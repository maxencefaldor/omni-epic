import ast
import os
import traceback
import logging
from time import sleep
import re
from textwrap import indent, dedent
import base64
from io import BytesIO
from PIL import Image
import mediapy

from openai import OpenAI
from openai import RateLimitError, APIConnectionError
import anthropic
import google.generativeai as genai

from embodied.envs.pybullet import PyBullet
from omni_epic.core import prompts, ParseError
from omni_epic.robots import robot_dict
from omni_epic.envs import EnvironmentError, test_env_halts, test_env
logger = logging.getLogger(__name__)


class FM:

	def __init__(self, config):
		self._config = config
		self._client_name = self._config.client
		self._model = self._config.model
		self._client = self._create_client(self._client_name, self._model)

	def _create_client(self, client_name, model):
		if client_name == "openai":
			return OpenAI()
		elif client_name == "anthropic":
			return anthropic.Anthropic()
		elif client_name == "google":
			return genai.GenerativeModel(model)

	def _create_prompt_multimodal(self, prompt, input_images):
		client_name = self._client_name
		prompt = prompt.strip()
		if client_name == "openai":
			new_prompt = [
				{
					"type": "image_url",
					"image_url": {
						"url": f"data:image/png;base64,{xs}",
						# "detail": "low",
					}
				} for xs in input_images
			]
			new_prompt.append({"type": "text", "text": prompt})
		elif client_name == "anthropic":
			new_prompt = [
				{
					"type": "image",
					"source": {
						"type": "base64",
						"media_type": "image/png",
						"data": xs,
					},
				} for xs in input_images
			]
			new_prompt.append({"type": "text", "text": prompt})
		elif client_name == "google":
			new_prompt = [Image.open(BytesIO(base64.b64decode(xs))) for xs in input_images]
			new_prompt.append(prompt)
		return new_prompt

	def _chat_completion(self, system_prompt, user_prompt):
		while True:
			try:
				if self._client_name == "openai":
					completion = self._client.chat.completions.create(
						messages=[
							{"role": "system", "content": system_prompt},
							{"role": "user", "content": user_prompt},
						],
						model=self._model,
						max_tokens=self._config.max_tokens,
						temperature=self._config.temperature,
					).choices[0].message.content
				elif self._client_name == "anthropic":
					completion = self._client.messages.create(
						system=system_prompt,
						messages=[
							{"role": "user", "content": user_prompt},
						],
						model=self._model,
						max_tokens=self._config.max_tokens,
						temperature=self._config.temperature,
					).content[0].text
				elif self._client_name == "google":
					# NOTE: have to use this complicated multiprocessing thing so that training with JAX runs after using the gemini API.
					from multiprocessing import Process, Queue

					def generate_content(q, user_prompt, system_prompt, client, config):
						user_prompt = user_prompt if isinstance(user_prompt, list) else [user_prompt]
						completion = client.generate_content(
							contents=[f"System prompt: {system_prompt}", *user_prompt],
							generation_config=genai.types.GenerationConfig(
								max_output_tokens=config.max_tokens,
								temperature=config.temperature,
							)
						).text
						q.put(completion)

					q = Queue()
					p = Process(target=generate_content, args=(q, user_prompt, system_prompt, self._client, self._config))
					p.start()
					p.join()  # Wait for the process to finish
					completion = q.get()  # Get the result from the queue
				# Log completion
				completion = completion.strip()
				logger.info({"system_prompt": system_prompt, "user_prompt": user_prompt, "completion": completion})
				return completion
			except (RateLimitError, APIConnectionError, Exception) as e:
				logger.info(f"API got error {e}. Retrying after 10 seconds.")
				sleep(10)

	def wrap_string(self, string):
		"""Wrap string in triple quotes."""
		return f"\"\"\"\n{string}\n\"\"\""

	def wrap_code(self, code):
		"""Wrap code in a python block."""
		return f"```python\n{code}\n```"

	def filter_error(self, error):
		error = error.strip()
		lines = []
		for line in error.splitlines():
			if set(line) == {'^', ' '}:
				pass
			else:
				lines.append(line)
		return '\n'.join(lines)

	def get_env_code(self, env_path):
		"""Get environment code from env_path."""
		env_code = open(env_path).read()
		env_code_wrapped = self.wrap_code(env_code.strip())
		return env_code_wrapped

	def get_env_codes(self, env_paths):
		"""Get environment codes from env_paths."""
		env_codes = [self.get_env_code(env_path) for env_path in env_paths]
		env_codes = "\n\n".join(env_codes)
		return env_codes

	def get_env_codes_example(self, robot):
		"""Get few-shot environment code examples for a given robot."""
		env_paths_example = robot_dict[robot]["env_paths_example"]
		return self.get_env_codes(env_paths_example)

	def parse_env_code(self, completion):
		"""Parse the environment code from the completion."""
		match = re.search(r"Environment code:\s*```python\s*(.*?)\s*```", completion, re.DOTALL | re.IGNORECASE)

		if match:
			env_code = match.group(1).strip()
			return env_code
		else:
			raise ParseError("No environment code found in your response. Please follow the desired format.")

	def parse_next_task_desc(self, completion):
		"""Parse the next task description from the completion."""
		match = re.search(r"Next task description:\s*\"\"\"(.*)\"\"\"", completion, re.DOTALL | re.IGNORECASE)

		if match:
			next_task_desc = dedent(match.group(1)).strip()
			return next_task_desc
		else:
			raise ParseError("No next task description found in your response. Please follow the desired format.")

	def parse_success(self, completion):
		"""Parse the task success from the completion."""
		match = re.search(r"Did the robot solve the task\?:\s*(.*?)$", completion, re.MULTILINE | re.IGNORECASE)

		if match:
			task_success = match.group(1).strip()
			task_success = task_success.split(" ")[0].lower()  # if there are words after the answer
			return "yes" in task_success  # handle the case where there are punctuations
		else:
			raise ParseError("No task success/failure evaluation found in your response. Please follow the desired format.")

	def parse_success_reasoning(self, completion):
		"""Parse the task success reasoning from the completion."""
		match = re.search(r"Reasoning for task success/failure:\s*(.*?)\s*Did the robot solve the task\?:", completion, re.DOTALL | re.IGNORECASE)

		if match:
			task_success_reasoning = match.group(1).strip()
			return task_success_reasoning
		else:
			raise ParseError("No task success/failure reasoning found in your response. Please follow the desired format.")

	def parse_interestingness(self, completion):
		"""Parse the interestingness from the completion."""
		match = re.search(r"Is the new task interesting\?:\s*(.*?)$", completion, re.MULTILINE | re.IGNORECASE)

		if match:
			is_interesting = match.group(1).strip()
			is_interesting = is_interesting.split(" ")[0].lower()  # if there are words after the answer
			return "yes" in is_interesting
		else:
			raise ParseError("No task interestingness evaluation found in your response. Please follow the desired format.")

	def query_env_code(self, robot, task_desc, add_examples=True, env_paths_other=[]):
		"""Query environment code for the given task description."""
		# Create prompts
		robot_desc = robot_dict[robot]["robot_desc"]
		task_desc_wrapped = self.wrap_string(task_desc)
		env_codes_example = [self.get_env_codes_example(robot)] if add_examples else []
		env_code_others = [self.get_env_codes(env_paths_other)] if env_paths_other else []
		env_codes_example = "\n\n".join(env_codes_example + env_code_others)

		system_prompt = prompts.query_env_code.system_prompt.format(ROBOT_DESC=robot_desc)
		user_prompt = prompts.query_env_code.user_prompt.format(ENV_CODES_EXAMPLE=env_codes_example, TASK_DESC=task_desc_wrapped)

		# Prompt FM
		logger.info(f"Query environment code.\nTask description:\n{task_desc_wrapped}")
		completion = self._chat_completion(system_prompt, user_prompt)
		return completion

	def reflect_error(self, robot, env_code, error, add_examples=True, env_paths_other=[]):
		"""Reflect on environment code error."""
		# Create prompts
		robot_desc = robot_dict[robot]["robot_desc"]
		env_code_wrapped = self.wrap_code(env_code)
		error_wrapped = self.wrap_string(error)
		env_codes_example = [self.get_env_codes_example(robot)] if add_examples else []
		env_codes_other = [self.get_env_codes(env_paths_other)] if env_paths_other else []
		env_codes_example = "\n\n".join(env_codes_example + env_codes_other)

		system_prompt = prompts.reflect_error.system_prompt.format(ROBOT_DESC=robot_desc)
		user_prompt = prompts.reflect_error.user_prompt.format(
			ENV_CODES_EXAMPLE=env_codes_example,
			ENV_CODE=env_code_wrapped,
			ERROR=error_wrapped,
		)

		# Prompt FM
		logger.info(f"Reflect on error for environment code.\nError:\n{error}")
		completion = self._chat_completion(system_prompt, user_prompt)
		return completion

	def iterate_on_errors(self, robot, task_desc, completion, task_path, add_examples=True, env_paths_other=[], iteration_max=5):
		os.makedirs(task_path, exist_ok=True)
		iteration = 0
		while iteration <= iteration_max:
			try:
				# Parse environment code
				env_code = self.parse_env_code(completion)

				# Save environment code before replacing docstring because it can raise an error if code is incorrect
				env_path = os.path.join(task_path, f"env_{iteration}.py")
				with open(env_path, "w") as f:
					f.write(env_code)

				# Replace docstring
				env_code = update_env_docstring(env_code, task_desc)

				# Save environment code
				env_path = os.path.join(task_path, f"env_{iteration}.py")
				with open(env_path, "w") as f:
					f.write(env_code)

				# Test if environment halts
				test_env_halts(env_path, timeout=10.)

				# Test environment
				test_env(env_path)
			except ParseError as e:
				env_code = str(None)
				error = str(e) + f"\n\"\"\"\n\nTask description:\n\"\"\"{task_desc}"
			except EnvironmentError as e:
				error = str(e)
			except Exception:
				error = traceback.format_exc()
				error = self.filter_error(error)
			else:
				logger.info(f"Generate environment code, iteration {iteration}: SUCESS")

				# Visualize environment
				env = PyBullet(env_path=env_path, vision=False)._env
				renders, renders3p = env.visualize()
				env.close()
				mediapy.write_video(os.path.join(task_path, "render.mp4"), renders)
				mediapy.write_video(os.path.join(task_path, "render3p.mp4"), renders3p)

				return iteration
			logger.info(f"Generate environment code, iteration {iteration}: ERROR")

			# Reflect on error
			completion = self.reflect_error(robot, env_code, error, add_examples=add_examples, env_paths_other=env_paths_other)
			iteration += 1
		return -1

	def generate_env_code(self, robot, task_desc, task_path, add_examples=True, env_paths_other=[], iteration_max=5):
		"""Generate environment code for the given task description."""
		# Query environment code
		completion = self.query_env_code(robot, task_desc)

		# Iterate on errors
		iteration = self.iterate_on_errors(
			robot, task_desc,
			completion,
			task_path,
			add_examples=add_examples,
			env_paths_other=env_paths_other,
			iteration_max=iteration_max
		)

		return iteration

	def reflect_task(self, robot, env_path, env_paths_other, add_examples=True):
		"""Reflect on task."""
		# Create prompts
		robot_desc = robot_dict[robot]["robot_desc"]
		env_code_wrapped = self.get_env_code(env_path)
		env_codes_example = [self.get_env_codes_example(robot)] if add_examples else []
		env_codes_other = [self.get_env_codes(env_paths_other)] if env_paths_other else []
		env_codes_example = "\n\n".join(env_codes_example + env_codes_other)

		system_prompt = prompts.reflect_task.system_prompt.format(ROBOT_DESC=robot_desc)
		user_prompt = prompts.reflect_task.user_prompt.format(
			ENV_CODES_EXAMPLE=env_codes_example,
			ENV_CODE=env_code_wrapped,
		)

		# Prompt FM
		logger.info(f"Reflect on task.")
		completion = self._chat_completion(system_prompt, user_prompt)
		return completion

	def reflect_task_with_vision(self, robot, env_path, env_paths_other, failure_reasoning, input_image, add_examples=True):
		"""Reflect on task with vision."""
		# Create prompts
		robot_desc = robot_dict[robot]["robot_desc"]
		env_code_wrapped = self.get_env_code(env_path)
		env_codes_example = [self.get_env_codes_example(robot)] if add_examples else []
		env_codes_other = [self.get_env_codes(env_paths_other)] if env_paths_other else []
		env_codes_example = "\n\n".join(env_codes_example + env_codes_other)
		input_image = input_image if isinstance(input_image, list) else [input_image]

		# Build prompts
		system_prompt = prompts.reflect_task_with_vision.system_prompt.format(ROBOT_DESC=robot_desc)
		user_prompt = prompts.reflect_task_with_vision.user_prompt.format(
			ENV_CODES_EXAMPLE=env_codes_example,
			ENV_CODE=env_code_wrapped,
			FAILURE_REASONING=failure_reasoning,
		)

		# Add input image to user prompt
		user_prompt = self._create_prompt_multimodal(user_prompt, input_image)

		# Prompt FM
		logger.info(f"Reflect on task with vision.")
		completion = self._chat_completion(system_prompt, user_prompt)
		return completion

	def get_next_task_desc(self, robot, env_paths_learned, env_paths_failed, add_examples=True):
		"""Get the next task."""
		# Create prompts
		robot_desc = robot_dict[robot]["robot_desc"]
		env_codes_example = self.get_env_codes_example(robot) if add_examples else "None"
		env_codes_learned = self.get_env_codes(env_paths_learned)
		env_codes_learned = "None" if env_codes_learned == "" else env_codes_learned
		env_codes_failed = self.get_env_codes(env_paths_failed)
		env_codes_failed = "None" if env_codes_failed == "" else env_codes_failed

		if self._config.enable_moi:
			system_prompt = prompts.query_next_task_desc.system_prompt
			user_prompt = prompts.query_next_task_desc.user_prompt
		else:
			system_prompt = prompts.query_next_task_desc_no_moi.system_prompt
			user_prompt = prompts.query_next_task_desc_no_moi.user_prompt
		system_prompt = system_prompt.format(ROBOT_DESC=robot_desc)
		user_prompt = user_prompt.format(ENV_CODES_EXAMPLE=env_codes_example, ENV_CODES_LEARNED=env_codes_learned, ENV_CODES_FAILED=env_codes_failed)

		# Prompt FM
		logger.info(f"Query next task description.")
		completion = self._chat_completion(system_prompt, user_prompt)

		# Parse next task description
		try:
			next_task_desc = self.parse_next_task_desc(completion)
		except ParseError as e:
			logger.info(f"Querying next task:\nError:\n{e}")
			return self.get_next_task_desc(robot, env_paths_learned, env_paths_failed)
		else:
			return next_task_desc

	def query_success_with_vision(
			self,
			robot, robot_desc,
			task_desc, task_codepath,
			input_image,
	):
		"""Evaluate the success of the task using vision input.

		Args:
			robot: Robot name.
			robot_desc: Robot description.
			task_desc: Task description.
			task_codepath: Task codepath.
			input_image: Input image. Can be one image or a list of images.

		Returns:
			completion: Completion text generated by FM.
			task_success: Task success/failure indicator.
			success_reasoning: Reasoning for task success/failure.
		"""
		# Process inputs
		task_desc = self.wrap_string(task_desc)
		env_code = self.get_env_codes([task_codepath])
		input_image = input_image if isinstance(input_image, list) else [input_image]

		# Build prompts
		system_prompt = prompts.query_success.system_prompt.format(ROBOT_DESC=robot_desc)
		user_prompt = prompts.query_success.user_prompt.format(ENV_CODE=env_code)
		user_prompt = self._create_prompt_multimodal(user_prompt, input_image)

		# Prompt FM
		system_prompt = system_prompt.strip()
		completion = self._chat_completion(system_prompt, user_prompt)

		# Parse success detection
		try:
			task_success = self.parse_success(completion)
		except Exception as e:
			logger.info(f"Error: {e} Trying again.")
			return self.query_success_with_vision(
				robot, robot_desc,
				task_desc, task_codepath,
				input_image,
			)

		# Parse success reasoning
		try:
			task_success_reasoning = self.parse_success_reasoning(completion)
		except Exception as e:
			logger.info(f"Error: {e} Trying again.")
			return self.query_success_with_vision(
				robot, robot_desc,
				task_desc, task_codepath,
				input_image,
			)

		return completion, task_success, task_success_reasoning

	def query_interestingness(self, robot_desc, query_codepath, compare_codepaths):
		"""Evaluate if the generated task is interesting by comparing with the given tasks.

		Args:
			robot_desc: Robot description.
			query_codepath: Query codepath.
			compare_codepaths: Compare codepaths.

		Returns:
			completion: Completion text generated by FM.
			is_interesting: Task interestingness indicator.
		"""
		# Process inputs
		target_code = self.get_env_codes([query_codepath])
		compare_codes = self.get_env_codes(compare_codepaths)

		# Build prompts
		system_prompt = prompts.query_interestingness.system_prompt.format(ROBOT_DESC=robot_desc)
		user_prompt = prompts.query_interestingness.user_prompt.format(ENV_CODES_EXAMPLE=compare_codes, ENV_CODE=target_code)

		# Query FM
		system_prompt = system_prompt.strip()
		user_prompt = user_prompt.strip()
		completion = self._chat_completion(system_prompt, user_prompt)

		# Parse interestingness
		try:
			is_interesting = self.parse_interestingness(completion)
		except Exception as e:
			logger.info(f"Error: {e} Trying again.")
			return self.query_interestingness(query_codepath, compare_codepaths)

		return completion, is_interesting


def update_env_docstring(env_code, task_desc):
	"""
	Modifies or adds a docstring to the class `Env` in the given env_code.
	
	Args:
	env_code (str): The environment code containing class `Env`.
	new_docstring (str): The new docstring to insert for the class 'Env'.
	
	Returns:
	str: The modified env_code if the class 'Env' is found, or unchanged code otherwise.
	"""
	indentation = re.search(r'\n([ \t]+)def get_task_rewards', env_code, re.DOTALL).group(1)
	task_desc_wrapped = '\n' + indent(task_desc, indentation) + '\n' + indentation

	class DocstringUpdater(ast.NodeTransformer):
		"""
		AST Node Transformer to update or add docstrings to the specified class 'Env'.
		"""
		def visit_ClassDef(self, node):
			if node.name == "Env":
				if not ast.get_docstring(node):
					# If no docstring, add one
					node.body.insert(0, ast.Expr(value=ast.Constant(value=task_desc_wrapped)))
				else:
					# Replace the existing docstring
					for i, n in enumerate(node.body):
						if isinstance(n, ast.Expr) and isinstance(n.value, (ast.Constant, ast.Constant)):
							node.body[i] = ast.Expr(value=ast.Constant(value=task_desc_wrapped))
							break
				return node
			return node

	# Parse the original code into an AST
	tree = ast.parse(env_code)

	# Modify the AST
	transformer = DocstringUpdater()
	modified_tree = transformer.visit(tree)

	# Convert the modified AST back to source code using ast.unparse
	modified_env_code = ast.unparse(modified_tree)

	return modified_env_code
