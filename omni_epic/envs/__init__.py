import multiprocessing
import numpy as np


# Task descriptions
task_dict = {
	"go_up_stairs_0.05m": """
Go up the stairs.

The stairs have 10 steps going up. The steps are 1.0 meters in length, 5.0 meters in width, and 0.05 meters in height. The robot starts on the ground, 2.0 meters away from the bottom of the stairs.
The task of the robot is to walk up the stairs, ensuring that it maintains its balance and does not fall.
""".strip(),

	"go_up_stairs_0.1m": """
Go up the stairs.

The stairs have 10 steps going up. The steps are 1.0 meters in length, 5.0 meters in width, and 0.1 meters in height. The robot starts on the ground, 2.0 meters away from the bottom of the stairs.
The task of the robot is to walk up the stairs, ensuring that it maintains its balance and does not fall.
""".strip(),

	"hurdle_0.05m": """
Run forward over one hurdle.

A hurdle is placed 2.0 meters ahead of the robot. The hurdle measures 0.1 meters in length, 5.0 meters in width, and 0.05 meters in height.
The task of the robot is to run forward, leaping over the hurdle without touching or sidestepping it.
""".strip(),

	"hurdles_0.05m": """
Run forward over a series of 5 hurdles.

A series of 5 hurdles spaced 2.0 meters apart, is placed 2.0 meters ahead of the robot. Each hurdle measures 0.1 meters in length, 5.0 meters in width, and 0.05 meters in height.
The task of the robot is to run forward, leaping over the hurdles without touching or sidestepping it.
""".strip(),

	"kick_ball_to_goal": """
Kick the ball towards the goal.

A red ball is placed 2.0 meters from the robot. The goal is represented by a green box and is placed 5.0 meters from the robot.
The task of the robot is to reach the ball as quickly as possible and kick the ball to push it towards the goal.
""".strip(),

	"kick_ball_to_goalposts": """
Kick the ball towards the goal.

A red ball is placed 2.0 meters away from the robot. The goal is represented by two green posts and is placed 5.0 meters from the robot. The goalposts are spaced at least twice the ball's diameter apart.
The task of the robot is to reach the ball as quickly as possible and kick the ball to push it towards the goal between the two goalposts.
""".strip(),

	"walk_backward_on_cylinder": """
Walk backward on a cylinder.

The robot is standing on a 2-meter-radius cylinder that can roll on the floor on the x axis.
The task of the robot is to walk backward on the cylinder while not falling off.
""".strip(),

	"dance_on_platform": """
Dance on a platform.

The robot is standing on a yellow platform that is 4.0 meters in length and width, and 0.1 meters in height above the ground.
The task of the robot is to move its body, to dance to a periodic rhythm.
""".strip(),

	"cross_bridge_gap_0.05m": """
Cross a pride-colored bridge with tiny gaps.

A 6-meter-long bridge with pride colors links the start platform to the end platform. The bridge has tiny gaps of 0.05 meters between each segment.
The task of the robot is to cross the bridge as quickly as possible.
""".strip(),

	"cross_bridge_gap_0.1m": """
Cross a pride-colored bridge with tiny gaps.

A 6-meter-long bridge with pride colors links the start platform to the end platform. The bridge has tiny gaps of 0.1 meters between each segment.
The task of the robot is to cross the bridge as quickly as possible.
""".strip(),
}


# Test Environment
terminated_error = """
The method Env.get_terminated returns True immediately following Env.reset, leading the episode to terminate prematurely.

Possible causes:
- The method Env.get_terminated might not be implemented correctly.
- The method Env.reset might not be implemented correctly, causing the termination condition to be met immediately after reset.

To fix:
- Check the implementation of Env.get_terminated and ensure that the logic is correct.
- Check the implementation of Env.reset and make sure that the termination condition is not met immediately after reset. For example, ensure that the initial state of the robot does not meet the termination condition after reset.
""".strip()

success_error = """
The method Env.get_success returns True immediately following Env.reset, leading to completing the task prematurely.

Possible causes:
- The method Env.get_success might not be implemented correctly.
- The method Env.reset might not be implemented correctly, causing the success condition to be met immediately after reset.

To fix:
- Check the implementation of Env.get_success and ensure that the logic is correct.
- Check the implementation of Env.reset and make sure that the success condition is not met immediately after reset.
	- Ensure that the initial state of the robot does not meet the success condition after reset.
""".strip()

robot_colliding_error = """
A collision has been detected between the robot and another body, immediately following Env.reset. This issue typically indicates a problem with the initial position of the robot relative to its environment, leading to overlaps.

Possible causes:
- The initial position of the robot might be set incorrectly during Env.reset.
- The initial position or orientation of at least one object might be set incorrectly during Env.reset.

To fix:
- Ensure that the robot's initial position is set relative to the platform it starts on, as demonstrated in the provided environment code examples. For example, if the robot starts on a platform, its initial position should be set to [self.platform_position[0], self.platform_position[1], self.platform_position[2] + self.platform_size[2] / 2 + self.robot.links["base"].position_init[2]].
- Check Env.reset and make sure that the initial position of the robot is set correctly.
	- Ensure that the initial x and y coordinates of the robot are set to the designated starting point of the supporting ground or platform to avoid off-edge placements.
	- Ensure that the initial z coordinate of the robot is set to a height that allows for safe clearance above the supporting ground or platform, avoiding any unintended collision with the surface.
- Check Env.reset and make sure that the initial position of the objects are set correctly.
	- Ensure that the initial position of each object is spaced far enough from the robot, taking into account the size and shape of each object to prevent overlapping.
	- Ensure that the initial orientation of each object is appropriate, and that any directional aspects of the objects do not interfere with the robot's starting position.
""".strip()

object_colliding_error = """
A collision has been detected between at least two bodies, immediately following Env.reset. This issue typically indicates a problem with the initial position or orientation of the different bodies, leading to overlaps.

To fix:
- Check Env.reset and make sure that the initial position and orientation of each object are set correctly.
	- If an object is supposed to be initialized on a supporting ground or platform, ensure that the initial x and y coordinates of the object are set to the designated starting point of the supporting ground or platform to avoid off-edge placements.
	- If an object is supposed to be initialized on a supporting ground or platform, ensure that the initial z coordinate of the object is set to a height that allows for safe clearance above the supporting ground or platform, avoiding any unintended collision with the surface.
- Ensure that objects are spaced far enough from each other, taking into account the size and shape of each object to prevent overlapping.
- Ensure that the initial orientation of each object is appropriate, and that any directional aspects of objects do not interfere with each other.
""".strip()

robot_falling_error = """
The robot is falling immediately following Env.reset.

Possible causes:
- The initial position of the robot might be set incorrectly during Env.reset, causing it to start off the edge of a platform or unsupported area.
- No supporting ground or platform for the robot to stand on has been created during Env.reset, causing the robot to free fall.
- A supporting ground of platform for the robot to stand on exists, but it might not be large enough or its initial position might be set incorrectly, leading to inadequate support.

To fix:
- Check Env.reset and make sure that the initial position of the robot is set correctly.
	- Verify that the robot is initialized at a safe and central position on the platform or ground. Check the x and y coordinates to ensure they center the robot adequately on the available space.
	- Ensure the z coordinate positions the robot firmly on the surface, without any part suspended in air.
- Confirm the existence and adequacy of the platform or ground:
	- Check that a platform or ground is created to support the robot.
	- Ensure that the platform or ground is of appropriate dimensions to accommodate the robot's size.
	- Adjust the initial position of the platform or ground, making sure it aligns correctly with the initial position of the robot.
	- Make sure that the platform or ground is steady and stable, providing a secure foundation for the robot.
""".strip()

timeout_error = """
A method in class Env exceeded the time limit while running.

Possible causes:
- A method might contain an infinite loop.
- A method might take an excessive amount of time to complete.

To fix:
Check the implementation of Env and ensure that all methods including Env.__init__ have proper termination conditions and don't contain infinite loops.
""".strip()

class EnvironmentError(Exception):
	pass

def test_env(env_path):
	# Test Env.__init__
	from embodied.envs.pybullet import PyBullet
	env = PyBullet(env_path=env_path, vision=False)._env

	try:
		# Test Env.reset
		observation = env.reset()
		if not isinstance(observation, np.ndarray):
			raise EnvironmentError(
				f"Expected observation from Env.reset to be a numpy.ndarray, but received type '{type(observation).__name__}'. "
				"Please ensure that observation from Env.reset returns a numpy.ndarray."
			)

		# Test robot collision after Env.reset
		if env.is_robot_colliding():
			raise EnvironmentError(robot_colliding_error)

		# Test Env.step
		observation, reward, terminated, truncated, info = env.step(0. * env.action_space.sample())

		if not isinstance(observation, np.ndarray):
			raise EnvironmentError(
				f"Expected observation from Env.step to be a numpy.ndarray, but received type '{type(observation).__name__}'. "
				"Please ensure that observation from Env.step returns a numpy.ndarray."
			)

		if not isinstance(terminated, bool) and not isinstance(terminated, np.bool_) and not (isinstance(terminated, np.ndarray) and terminated.dtype == bool):
			raise EnvironmentError(
				f"Expected terminated from Env.step to be a boolean, but received type '{type(terminated).__name__}'. "
				"Please ensure that terminated from Env.step returns a boolean."
			)

		# Test Env.get_success
		success = env.get_success()
		if not isinstance(success, bool) and not isinstance(success, np.bool_) and not (isinstance(success, np.ndarray) and success.dtype == bool):
			raise EnvironmentError(
				f"Expected success from Env.get_success to be a boolean, but received type '{type(success).__name__}'. "
				"Please ensure that success from Env.get_success returns a boolean."
			)

		# Test robot collision after one Env.step call
		if env.is_robot_colliding():
			raise EnvironmentError(robot_colliding_error)

		# Test terminated after one Env.step call
		if terminated:
			raise EnvironmentError(terminated_error)

		# Test success after one Env.step call
		if success:
			raise EnvironmentError(success_error)

		for _ in range(100):
			env.step(0. * env.action_space.sample())
			if env.is_object_colliding():
				raise EnvironmentError(object_colliding_error)
			if env.is_robot_falling():
				raise EnvironmentError(robot_falling_error)
	except Exception as e:
		raise e
	finally:
		env.close()

def env_run_all(env_path):
	# Test Env.__init__
	from embodied.envs.pybullet import PyBullet
	env = PyBullet(env_path=env_path, vision=False)._env

	try:
		# Test Env.reset
		env.reset()

		# Test Env.step
		env.step(env.action_space.sample())

		# Test Env.get_success
		env.get_success()
	except:
		pass
	finally:
		env.close()

def test_env_halts(env_path, timeout=10.):
	process = multiprocessing.Process(target=env_run_all, args=(env_path,))
	process.start()

	process.join(timeout)
	if process.is_alive():
		process.terminate()
		process.join()
		raise EnvironmentError(timeout_error)


if __name__ == "__main__":
	env_path = "/workspace/src/env_not_halting.py"
	# env_path = "/workspace/src/env_error.py"
	# env_path = "/workspace/src/env_good.py"

	test_env_halts(env_path)
