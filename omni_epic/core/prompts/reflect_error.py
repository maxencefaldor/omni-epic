system_prompt = """
You are an expert in Python programming and reinforcement learning. Your goal is to implement an environment in PyBullet specifically designed to train a robot for a given task. You will be provided with environment code examples, with an environment code that returns an error when executed and with the specific error that was encountered. Your objective is to reason about the error and provide a new, improved environment code with no error.

Instructions:
- Write code without using placeholders.
- Don't change the import statements.
- For each object, always define its size first, and ensure the object's initial position is set relative to the platform it starts on or any other object, as demonstrated in the provided environment code examples. For example, if an object is initialized on the ground, define its position as: [self.platform_position[0], self.platform_position[1], self.platform_position[2] + self.platform_size[2] / 2 + self.object_size[2] / 2].
- Ensure the robot's initial position is set relative to the platform it starts on, as demonstrated in the provided environment code examples. For example, if the robot starts on a platform, its initial position should be set to [self.platform_position[0], self.platform_position[1], self.platform_position[2] + self.platform_size[2] / 2 + self.robot.links["base"].position_init[2]].
- Implement the methods `Env.reset()`, `Env.step()`, `Env.get_task_rewards()`, `Env.get_terminated()`, `Env.get_success()`. You can implement additional methods if needed.
- `Env.get_task_rewards()` returns a dictionary with the different reward components to help the robot learn the task. You should implement dense reward components that are easy to optimize and defined in the range -10. to 10.
- `Env.get_terminated()` returns a boolean that indicates whether the episode is terminated.
- `Env.get_success()` returns a boolean that indicates whether the task is successfully completed.

Robot description:
{ROBOT_DESC}

Desired format:
How to solve the error:
<reasoning>

Environment code:
```python
<environment code>
```
""".strip()

user_prompt = """
Environment code examples:
{ENV_CODES_EXAMPLE}

Environment code with error:
{ENV_CODE}

Error:
{ERROR}
""".strip()
