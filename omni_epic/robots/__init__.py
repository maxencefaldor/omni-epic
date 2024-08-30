from textwrap import dedent


robot_dict = {
	"ant": {
		"robot_desc": dedent("""
			Quadruped robot consisting of a base and four articulated legs.
			- The links of the robot, given by `robot.links.keys()`, include but are not limited to `['base', 'front_left_leg', 'front_left_foot', 'front_right_leg', 'front_right_foot', 'left_back_leg', 'left_back_foot', 'right_back_leg', 'right_back_foot']`.
			- The robot measures 2 m in width and 1 m in height.
			- The initial position of the robot is given by `robot.links["base"].position_init`, and is appropriate to position the robot on a platform whose surface is at z = 0.
			- The initial orientation of the robot is given by `robot.links["base"].orientation_init`, which aligns the robot to face toward the positive x-axis.
			""").strip(),
		"env_paths_example": [
			"/workspace/src/omni_epic/envs/ant/balance_board.py",
			"/workspace/src/omni_epic/envs/ant/cross_bridge.py",
			"/workspace/src/omni_epic/envs/ant/cross_lava.py",
			"/workspace/src/omni_epic/envs/ant/go_down_stairs.py",
			"/workspace/src/omni_epic/envs/ant/go_to_box.py",
			"/workspace/src/omni_epic/envs/ant/kick_ball.py",
			# "/workspace/src/omni_epic/envs/ant/maze.py",
			"/workspace/src/omni_epic/envs/ant/open_door.py",
			# "/workspace/src/omni_epic/envs/ant/go_forward.py",
			"/workspace/src/omni_epic/envs/ant/walk_on_cylinder.py",
		],
		"task_descs_init": [
			dedent("""
			Cross a pride-colored bridge with gaps to reach a platform.

			Description:
			- A start platform and an end platform (each 3 m in size and 0.5 m in thickness) are placed 30 m apart.
			- The two platforms are connected by a bridge (2 m wide) divided in multiple segments. Each segment has a different color corresponding to the pride colors.
			- The segments are separated by gaps measuring 0.1 m.
			The robot is initialized on the start platform.
			The task of the robot is to cross the bridge to reach the end platform as fast as possible.

			Success:
			The task is successfully completed when the robot reaches the end platform.

			Rewards:
			To help the robot complete the task:
			- The robot receives a reward for each time step it remains standing on the bridge or platforms, encouraging steady progress.
			- The robot is rewarded based on how much it reduces the distance to the end platform, incentivizing swift movement towards the goal.

			Termination:
			The task terminates immediately if the robot falls off the start platform, any segment of the bridge, or the end platform.
			""").strip(),
			dedent("""
			Go backward on top of a rolling cylinder.

			Description:
			- The environment consists of a large flat ground measuring 1000 x 1000 x 10 m.
			- A cylinder with a radius of 2 m and a height of 3 m is placed on the ground and can roll along the x-axis.
			- The cylinder's initial position is at the center of the ground, and it is oriented to roll along the x-axis.
			- The robot is initialized on top of the cylinder.
			- The task of the robot is to go backward while balancing on top of the rolling cylinder.

			Success:
			The task is completed if the robot rolls more than 5 m backward without falling off.

			Rewards:
			To guide the robot to complete the task:
			- The robot receives a reward for each time step it remains balanced on the cylinder.
			- The robot receives a reward for backward velocity along the x-axis.

			Termination:
			The task terminates immediately if the is not standing on the cylinder or if the robot falls off the cylinder.
			""").strip(),
			dedent("""
			Dodge flying balls.

			Description:
			- The environment is a square arena measuring 20 x 20 x 5 m.
			- The robot is initialized at the center of the arena.
			- Every second a ball is launched toward the robot at varying speeds from random positions around the arena.
			- The task of the robot is to avoid being hit by the balls while remaining within the arena boundaries.

			Success:
			The task is successfully completed if the robot dodges all the balls.

			Rewards:
			To help the robot complete the task:
			- The robot is rewarded for standing at each time step.

			Termination:
			The task terminates immediately if the robot is hit by a ball or if the robot falls off the arena.
			""").strip()
		]
	},

	"humanoid": {
		"robot_desc": dedent("""
			Humanoid robot consisting of a base with two legs, two arms and a head.
			- The links of the robot, given by `robot.links.keys()`, include but are not limited to `['base', 'lwaist', 'pelvis', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot', 'right_upper_arm', 'right_lower_arm', 'left_upper_arm', 'left_lower_arm']`.
			- The robot measures 0.5 m in width and 1.8 m in height.
			- The initial position of the robot is given by `robot.links["base"].position_init`, and is appropriate to position the robot on a platform whose surface is at z = 0.
			- The initial orientation of the robot is given by `robot.links["base"].orientation_init`, which aligns the robot to face toward the positive x-axis.
			""").strip(),
		"env_paths_example": [
			"/workspace/src/omni_epic/envs/humanoid/balance_board.py",
			"/workspace/src/omni_epic/envs/humanoid/cross_bridge.py",
			"/workspace/src/omni_epic/envs/humanoid/cross_lava.py",
			"/workspace/src/omni_epic/envs/humanoid/go_down_stairs.py",
			"/workspace/src/omni_epic/envs/humanoid/go_to_box.py",
			"/workspace/src/omni_epic/envs/humanoid/kick_ball.py",
			# "/workspace/src/omni_epic/envs/humanoid/maze.py",
			"/workspace/src/omni_epic/envs/humanoid/open_door.py",
			# "/workspace/src/omni_epic/envs/humanoid/go_forward.py",
			"/workspace/src/omni_epic/envs/humanoid/walk_on_cylinder.py",
		],
		"task_descs_init": [
			dedent("""
			Cross a pride-colored bridge with gaps to reach a platform.

			Description:
			- A start platform and an end platform (each 3 m in size and 0.5 m in thickness) are placed 30 m apart.
			- The two platforms are connected by a bridge (2 m wide) divided in multiple segments. Each segment has a different color corresponding to the pride colors.
			- The segments are separated by gaps measuring 0.1 m.
			The robot is initialized on the start platform.
			The task of the robot is to cross the bridge to reach the end platform as fast as possible.

			Success:
			The task is successfully completed when the robot reaches the end platform.

			Rewards:
			To help the robot complete the task:
			- The robot receives a reward for each time step it remains standing on the bridge or platforms, encouraging steady progress.
			- The robot is rewarded based on how much it reduces the distance to the end platform, incentivizing swift movement towards the goal.

			Termination:
			The task terminates immediately if the robot falls off the start platform, any segment of the bridge, or the end platform.
			""").strip(),
			dedent("""
			Go backward on top of a rolling cylinder.

			Description:
			- The environment consists of a large flat ground measuring 1000 x 1000 x 10 m.
			- A cylinder with a radius of 2 m and a height of 3 m is placed on the ground and can roll along the x-axis.
			- The cylinder's initial position is at the center of the ground, and it is oriented to roll along the x-axis.
			- The robot is initialized on top of the cylinder.
			- The task of the robot is to go backward while balancing on top of the rolling cylinder.

			Success:
			The task is completed if the robot rolls more than 5 m backward without falling off.

			Rewards:
			To guide the robot to complete the task:
			- The robot receives a reward for each time step it remains balanced on the cylinder.
			- The robot receives a reward for backward velocity along the x-axis.

			Termination:
			The task terminates immediately if the is not standing on the cylinder or if the robot falls off the cylinder.
			""").strip(),
			dedent("""
			Dodge flying balls.

			Description:
			- The environment is a square arena measuring 20 x 20 x 5 m.
			- The robot is initialized at the center of the arena.
			- Every second a ball is launched toward the robot at varying speeds from random positions around the arena.
			- The task of the robot is to avoid being hit by the balls while remaining within the arena boundaries.

			Success:
			The task is successfully completed if the robot dodges all the balls.

			Rewards:
			To help the robot complete the task:
			- The robot is rewarded for standing at each time step.

			Termination:
			The task terminates immediately if the robot is hit by a ball or if the robot falls off the arena.
			""").strip()
		]
	},

	"r2d2": {
		"robot_desc": dedent("""
			R2D2 robot that can roll on wheels up to 10 m/s and jump 1 m high.
			- The robot measures 0.5 m in width and 1 m in height.
			- The initial position of the robot is given by `robot.links["base"].position_init`, and is appropriate to position the robot on a platform whose surface is at z = 0.
			- The initial orientation of the robot is given by `robot.links["base"].orientation_init`, which aligns the robot to face toward the positive x-axis.
			""").strip(),
		"env_paths_example": [
			"/workspace/src/omni_epic/envs/r2d2/balance_board.py",
			"/workspace/src/omni_epic/envs/r2d2/cross_bridge.py",
			"/workspace/src/omni_epic/envs/r2d2/cross_lava.py",
			"/workspace/src/omni_epic/envs/r2d2/go_down_stairs.py",
			"/workspace/src/omni_epic/envs/r2d2/go_to_box.py",
			"/workspace/src/omni_epic/envs/r2d2/kick_ball.py",
			# "/workspace/src/omni_epic/envs/r2d2/maze.py",
			"/workspace/src/omni_epic/envs/r2d2/open_door.py",
			# "/workspace/src/omni_epic/envs/r2d2/go_forward.py",
			"/workspace/src/omni_epic/envs/r2d2/walk_on_cylinder.py",
		],
		"task_descs_init": [
			dedent("""
			Cross a pride-colored bridge with gaps to reach a platform.

			Description:
			- A start platform and an end platform (each 3 m in size and 0.5 m in thickness) are placed 50 m apart.
			- The two platforms are connected by a bridge (2 m wide) divided in multiple segments. Each segment has a different color corresponding to the pride colors.
			- The segments are separated by gaps measuring 2 m.
			The robot is initialized on the start platform.
			The task of the robot is to cross the bridge to reach the end platform as fast as possible.

			Success:
			The task is successfully completed when the robot reaches the end platform.

			Rewards:
			To help the robot complete the task:
			- The robot receives a reward for each time step it remains standing on the bridge or platforms, encouraging steady progress.
			- The robot is rewarded based on how much it reduces the distance to the end platform, incentivizing swift movement towards the goal.

			Termination:
			The task terminates immediately if the robot falls off the start platform, any segment of the bridge, or the end platform.
			""").strip(),
			dedent("""
			Ascend a series of stairs to reach a platform.

			Description:
			- The environment consists of a ground platform (1000 m x 10 m x 10 m) and a set of 10 steps.
			- Each step has dimensions of 1 m in length, 10 m in width, and 0.2 m in height.
			- The steps are positioned to form an ascending staircase, with each subsequent step higher than the previous one.
			The robot is initialized on the ground at the bottom of the stairs.

			Success:
			The task is completed when the robot successfully ascends the stairs and reaches the top platform.

			Rewards:
			To help the robot complete the task:
			- The robot is rewarded for survival at each time step.
			- The robot is rewarded for forward velocity, incentivizing it to move up the stairs.

			Termination:
			The task terminates immediately if the robot falls off the stairs or the top platform.
			""").strip(),
			dedent("""
			Kick a ball into a goal.

			Description:
			- The environment consists of a large flat ground measuring 1000 x 1000 x 10 meters.
			- A ball with a radius of 0.5 meters is placed randomly on the ground.
			- The goal is defined by two goal posts, each 2 meters high and placed 3 meters apart, forming a goal area.
			- The robot is initialized at a fixed position on the ground.
			- The task of the robot is to move across the ground, reach the ball, and kick it into the goal.

			Success:
			The task is successfully completed if the robot kicks the ball so that it passes between the two goal posts.

			Rewards:
			To help the robot complete the task:
			- The robot is rewarded for survival at each time step.
			- The robot is rewarded for decreasing its distance to the ball.
			- The robot is rewarded for kicking the ball towards the goal, with additional rewards for successfully kicking the ball into the goal.

			Termination:
			The task does not have a specific termination condition.
			""").strip()
		]
	},
}
