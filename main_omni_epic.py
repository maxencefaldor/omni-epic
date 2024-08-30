import copy
import os
import math
import numpy as np
import re
import json
import hydra
from omegaconf import DictConfig

from omni_epic.robots import robot_dict
from omni_epic.core.fm import FM
from main_dreamer import main_dreamer
from run_utils import (
	get_images_from_video,
	save_images,
	encode_image,
	get_task_success_from_folder,
	parse_task_desc_from_env_code,
)
from rag_utils import get_similar_codepaths


def init_archive(archive_from_ckpt):
	archive_codepaths = []  # tasks that were successfully generated and trained
	archive_failedgens = []  # tasks that failed to generate compilable code
	archive_failedint = []  # tasks that failed interestingness eval
	archive_failedtrain = []  # tasks that failed to train a successful agent
	if len(archive_from_ckpt) > 0:
		# Initialize archive from checkpoint
		with open(archive_from_ckpt, 'r') as f:
			content = f.read()
			json_str = re.split('(?<=})\n(?={)', content)[-1]
			json_obj = json.loads(json_str)
			archive_codepaths = json_obj["codepaths"]
			archive_failedgens = json_obj["failedgens"]
			archive_failedint = json_obj["failedint"]
			archive_failedtrain = json_obj["failedtrain"]
	return archive_codepaths, archive_failedgens, archive_failedint, archive_failedtrain


@hydra.main(version_base=None, config_path="configs/", config_name="omni_epic")
def main(config: DictConfig):
	robot = config.robot
	robot_desc = robot_dict[robot]["robot_desc"]
	task_key_base = 'task'
	add_examples = config.add_examples

	# Create archive
	task_descs_init = robot_dict[robot]["task_descs_init"]
	archive_codepaths, archive_failedgens, archive_failedint, archive_failedtrain = init_archive(config.archive_from_ckpt)
	init_archive_size = len(task_descs_init)
	prev_num_iterations = len(archive_codepaths) + len(archive_failedgens) + len(archive_failedint) + len(archive_failedtrain)

	# Configs for each component
	config_task_generator = config.task_generator
	config_env_generator = config.environment_generator
	config_moi = config.model_of_interestingness
	config_success_detector = config.success_detector
	config_dreamer = config.dreamer
	config_task_iterator = config.task_iterator
	if config_success_detector.use_vision:
		config_task_iterator = config.task_iterator_vision
	num_steps_per_task = config.dreamer.run.steps

	# FM instance for each component
	fm_task_generator = FM(config_task_generator)
	fm_env_generator = FM(config_env_generator)
	fm_moi = FM(config_moi)
	fm_success_detector = FM(config_success_detector)
	fm_task_iterator = FM(config_task_iterator)

	# Variables to keep track of the iteration
	iterate_same_task = False
	iterate_same_task_count = 0
	iterations_spent_on_init_tasks = 0  # iterations spent on generating tasks from task_descs_init this run
	taskgen_choose_probs = np.ones(len(archive_codepaths))  # probability of choosing a task from the archive
	stop_iteration = False  # stop iterations, only used when config.iterate_until_success_gen is True

	# Override kwargs
	override_vars = config.override_vars
	taskgen_choose_probs = override_vars.get('taskgen_choose_probs', taskgen_choose_probs)
	taskgen_choose_probs = np.array(taskgen_choose_probs)
	iterate_same_task = override_vars.get('iterate_same_task', iterate_same_task)
	task_desc = override_vars.get('task_description', None)
	task_envpath = override_vars.get('task_envpath', None)

	prev_taskgen_choose_probs = copy.copy(taskgen_choose_probs)
	for iteration in range(config.iterations):
		if stop_iteration:
			break
		# Variables to keep track of the iteration
		iteration += prev_num_iterations  # add the number of iterations from the previous run
		task_key = f'{task_key_base}_{iteration}'
		task_dir = os.path.join(config.logdir, f'{task_key}')
		metadata = {}

		taskgen_example_paths = copy.copy(archive_codepaths)
		if iteration < len(task_descs_init):
			# First few iterations used to create tasks from seeded task descriptions
			task_desc = task_descs_init[iteration]
			taskgen_completion = fm_env_generator.query_env_code(robot, task_desc)
			metadata["init_task_desc"] = task_desc
			iterations_spent_on_init_tasks += 1
		elif iterate_same_task:
			# Iterate on the same task
			if not config.use_archive:
				taskgen_example_paths = []
			elif config_task_iterator.num_examples > 0 and len(archive_codepaths) > config_task_iterator.num_examples:
				# Find similar codepaths to the current task
				taskgen_example_paths, _ = get_similar_codepaths(
					task_envpath,
					archive_codepaths,
					num_returns=config_task_iterator.num_examples,
					embedding_method=config.embedding_method,
				)
			if config_success_detector.use_vision:
				# With vision
				taskgen_completion = fm_task_iterator.reflect_task_with_vision(
					robot,
					task_envpath,
					taskgen_example_paths,
					success_reasoning, input_image,  # should have been initialized in the previous iteration
					add_examples=add_examples,
				)
			else:
				# Wtihout vision
				taskgen_completion = fm_task_iterator.reflect_task(
					robot,
					task_envpath,
					taskgen_example_paths,
					add_examples=add_examples,
				)
			try:  # Get new task description if generated, otherwise use the previous one
				task_desc = parse_task_desc_from_env_code(taskgen_completion)
			except:
				pass
			iterate_same_task = False
			iterate_same_task_count += 1
			metadata["taskit_from_paths"] = [task_envpath]
			metadata["taskit_example_paths"] = taskgen_example_paths
			metadata["iterate_same_task_count"] = iterate_same_task_count
		else:
			if not config.use_archive:
				taskgen_example_paths = []
				taskgen_failed_paths = []
				taskgen_add_example_paths = []
			else:
				# Using prior knowledge that seeded task descs are very diverse, so adaptive num_examples
				num_examples = config_task_generator.num_examples
				num_examples = min(num_examples, max(math.ceil((iteration+1-init_archive_size) / init_archive_size), 1))
				num_examples_total = num_examples + config_task_generator.num_add_examples
				taskgen_add_example_paths = []
				# Choose examples to be fed into the prompt to generate the next task
				probs = taskgen_choose_probs if np.any(taskgen_choose_probs) else np.ones(len(archive_codepaths))
				probs /= np.sum(probs)
				chosen_idx = np.random.choice(len(archive_codepaths), p=probs)
				chosen_codepath = archive_codepaths[chosen_idx]
				if config_task_generator.num_examples > 0 and len(archive_codepaths) > num_examples:
					taskgen_example_paths, taskgen_example_indices = get_similar_codepaths(
						chosen_codepath,
						archive_codepaths,
						num_returns=num_examples_total,
						embedding_method=config.embedding_method,
					)
					prev_taskgen_choose_probs = copy.copy(taskgen_choose_probs)
					taskgen_choose_probs += 1  # Update counters for choosing examples
					taskgen_choose_probs[taskgen_example_indices] = 0
					taskgen_add_example_paths = taskgen_example_paths[num_examples:]
					taskgen_example_paths = taskgen_example_paths[:num_examples]
				# Choose failed examples to be fed into the prompt to generate the next task
				taskgen_failed_paths = copy.copy(archive_failedtrain)
				num_failed_examples = config_task_generator.num_failed_examples
				if num_failed_examples > 0 and len(archive_failedtrain) > num_failed_examples:
					taskgen_failed_paths, _ = get_similar_codepaths(
						chosen_codepath,
						archive_failedtrain,
						num_returns=num_failed_examples,
						embedding_method=config.embedding_method,
					)

			# Generate the next task
			metadata["taskgen_example_paths"] = taskgen_example_paths
			metadata["taskgen_failed_paths"] = taskgen_failed_paths
			metadata["taskgen_add_example_paths"] = taskgen_add_example_paths

			# Get the next task description
			task_desc = fm_task_generator.get_next_task_desc(
				robot,
				taskgen_example_paths,
				taskgen_failed_paths,
				add_examples=True,
			)

			# Query environment code
			taskgen_completion = fm_env_generator.query_env_code(robot, task_desc, add_examples=add_examples, env_paths_other=taskgen_example_paths + taskgen_add_example_paths)

		# Iterate on compilation errors for a max number of gens
		gen_num = fm_env_generator.iterate_on_errors(
			robot,
			task_desc,
			taskgen_completion,
			task_dir,
			add_examples=add_examples,
			env_paths_other=taskgen_example_paths,
			iteration_max=config.error_max_iterations,
		)

		# If generation was successful
		if gen_num >= 0:
			# Save the generated task envpath
			task_envpath = os.path.abspath(os.path.join(task_dir, f'env_{gen_num}.py'))
			metadata['envpath'] = task_envpath

			# Evaluate interestingness of the generated task
			if config.use_archive and config.enable_moi and len(archive_codepaths) > iterations_spent_on_init_tasks:
				# Evaluate whether the generated task is interesting by comparing with N most similar tasks
				moi_example_paths = copy.copy(archive_codepaths)
				if config_moi.num_examples > 0 and len(moi_example_paths) > config_moi.num_examples:
					moi_example_paths, _ = get_similar_codepaths(
						task_envpath,
						archive_codepaths,
						num_returns=config_moi.num_examples,
						embedding_method=config.embedding_method,
					)
				_, is_interesting = fm_moi.query_interestingness(
					robot_desc, task_envpath, moi_example_paths,
				)
				metadata['moi_example_paths'] = moi_example_paths
				metadata['is_interesting'] = is_interesting
			else:
				# Assume generated task is interesting
				is_interesting = True

			if is_interesting:
				if config.train_agent:
					# Train agent on the generated task
					dreamer_dir = os.path.join(task_dir, 'dreamer/')
					config_dreamer.logdir = dreamer_dir
					config_dreamer.env.path = task_envpath
					# If archive is not empty
					# and not first few iterations used to create tasks from seeded task descriptions
					if config.train_from_ckpt and len(archive_codepaths) > 0  \
						and not iteration < len(task_descs_init):
						ckpt_paths, _ = get_similar_codepaths(
							task_envpath,
							archive_codepaths,
							num_returns=1,
							embedding_method=config.embedding_method,
						)
						ckpt_path = ckpt_paths[0]
						ckpt_dir = os.path.join(os.path.dirname(ckpt_path), 'dreamer/')
						config_dreamer.run.from_checkpoint = os.path.join(ckpt_dir, 'checkpoint.ckpt')
						with open(os.path.join(ckpt_dir, 'metrics.jsonl'), 'r') as f:
							for line in f:
								ckpt_steps = json.loads(line)['step']
						config_dreamer.run.steps = ckpt_steps + num_steps_per_task
						metadata['train_from_ckpt'] = config_dreamer.run.from_checkpoint

					# Run Dreamer
					main_dreamer(config_dreamer)

					# Evaluate if the trained agent has successfully completed the task
					render_dir = os.path.join(dreamer_dir, 'eval')
					if config.enable_sd and config_success_detector.use_vision:
						# Use VLM to evaluate task success
						imagedir = os.path.join(task_dir, 'input_images/')
						video_files = [f for f in os.listdir(render_dir) if f.endswith('.mp4') and f.startswith('render')]
						video_file = os.path.join(render_dir, video_files[0])
						images = get_images_from_video(video_file)
						save_images(images, imagedir)
						input_image = encode_image(os.path.join(imagedir, "concat_image.png"))
						_, task_success, success_reasoning = fm_success_detector.query_success_with_vision(
							robot, robot_desc, task_desc, task_envpath, input_image,
						)
					elif config.enable_sd and not config_success_detector.use_vision:
						# Get task success from saved files
						task_success = get_task_success_from_folder(render_dir)
					else:
						task_success = True
				else:
					# Do not train agent and assume the task has succeeded
					task_success = True

				# If task is successful, add task to archive, else iterate on the same task
				metadata['task_success'] = task_success
				if task_success:
					# Add task to the archive
					archive_codepaths.append(task_envpath)
					iterate_same_task_count = 0
					taskgen_choose_probs = np.append(taskgen_choose_probs, 0)
					prev_taskgen_choose_probs = copy.copy(taskgen_choose_probs)
					if config.iterate_until_success_gen:
						stop_iteration = True
				else:
					# Iterate on the same task the next iteration
					if iterate_same_task_count < config_task_iterator.max_iterations:
						iterate_same_task = True
					else:
						iterate_same_task_count = 0
					archive_failedtrain.append(task_envpath)

			else:
				# If task is not interesting, add the task to the reject archive
				archive_failedint.append(task_envpath)
		else:
			# If generation failed, add the task to the reject archive
			archive_failedgens.append(task_dir)
			taskgen_choose_probs = prev_taskgen_choose_probs  # Reset taskgen_choose_probs

		# Save metadata about the task
		with open(os.path.join(task_dir, 'metadata.json'), 'w') as f:
			json.dump(metadata, f, indent=4)

		# Save the archive
		with open(os.path.join(config.logdir, 'archive.jsonl'), 'a') as f:
			f.write(json.dumps({
				'codepaths': archive_codepaths,
				'failedgens': archive_failedgens,
				'failedint': archive_failedint,
				'failedtrain': archive_failedtrain,
			}, indent=4) + '\n')

	return {
		'taskgen_choose_probs': taskgen_choose_probs,
	}


if __name__ == "__main__":
	main()
