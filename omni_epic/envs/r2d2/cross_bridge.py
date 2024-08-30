import numpy as np
from omni_epic.envs.r2d2.base import R2D2Env


class Env(R2D2Env):
    """
    Cross a pride-colored bridge to reach a platform.

    Description:
    - A start platform and an end platform (each 3 m in size and 0.5 m in thickness) are placed 30 m apart.
    - The two platforms are connected by a bridge (2 m wide) divided in multiple segments. Each segment has a different color corresponding to the pride colors.
    The robot is initialized on the start platform.
    The task of the robot is to cross the bridge to reach the end platform as fast as possible.

    Success:
    The task is successfully completed when the robot reaches the end platform.

    Rewards:
    To help the robot complete the task:
    - The robot receives a reward for each time step it remains on the bridge or platforms, encouraging steady progress.
    - The robot is rewarded based on how much it reduces the distance to the end platform, incentivizing swift movement towards the goal.

    Termination:
    The task terminates immediately if the robot falls off the start platform, any segment of the bridge, or the end platform.
    """

    def __init__(self):
        super().__init__()

        # Init start platform
        self.platform_size = [3., 3., 0.5]
        self.platform_start_position = [0., 0., 0.]
        self.platform_end_position = [self.platform_start_position[0] + 30., self.platform_start_position[1], self.platform_start_position[2]]
        self.platform_start_id = self.create_box(mass=0., half_extents=[self.platform_size[0] / 2, self.platform_size[1] / 2, self.platform_size[2] / 2], position=self.platform_start_position, color=[0.8, 0.8, 0.8, 1.])
        self.platform_end_id = self.create_box(mass=0., half_extents=[self.platform_size[0] / 2, self.platform_size[1] / 2, self.platform_size[2] / 2], position=self.platform_end_position, color=[0.8, 0.8, 0.8, 1.])

        # Init bridge
        self.bridge_length = self.platform_end_position[0] - self.platform_start_position[0] - self.platform_size[0]
        self.bridge_width = 2.
        pride_colors = [
            [1.0, 0.0, 0.0, 1.],  # Red
            [1.0, 0.5, 0.0, 1.],  # Orange
            [1.0, 1.0, 0.0, 1.],  # Yellow
            [0.0, 0.5, 0.0, 1.],  # Green
            [0.0, 0.0, 1.0, 1.],  # Blue
            [0.7, 0.0, 1.0, 1.],  # Violet
        ]

        # Segment length
        num_colors = len(pride_colors)
        segment_size = self.bridge_length / num_colors

        # Create segments
        for i, color in enumerate(pride_colors):
            segment_id = self.create_box(mass=0., half_extents=[segment_size / 2, self.bridge_width / 2, self.platform_size[2] / 2], position=[self.platform_start_position[0] + self.platform_size[0] / 2 + segment_size / 2 + i * segment_size, self.platform_start_position[1], self.platform_start_position[2]], color=color)
            self._p.changeDynamics(bodyUniqueId=segment_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

    def create_box(self, mass, half_extents, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def get_object_position(self, object_id):
        return np.asarray(self._p.getBasePositionAndOrientation(object_id)[0])

    def get_distance_to_object(self, object_id):
        object_position = self.get_object_position(object_id)
        robot_position = self.robot.links["base"].position
        return np.linalg.norm(object_position[:2] - robot_position[:2])

    def reset(self):
        observation = super().reset()

        # Reset robot position on start platform
        self._p.resetBasePositionAndOrientation(self.robot.robot_id, [self.platform_start_position[0], self.platform_start_position[1], self.platform_start_position[2] + self.platform_size[2] / 2 + self.robot.links["base"].position_init[2]], self.robot.links["base"].orientation_init)

        return observation

    def step(self, action):
        # Before taking action
        self.distance_to_platform_end = self.get_distance_to_object(self.platform_end_id)

        observation, reward, terminated, truncated, info = super().step(action)

        return observation, reward, terminated, truncated, info

    def get_task_rewards(self, action):
        # After taking action
        new_distance_to_platform_end = self.get_distance_to_object(self.platform_end_id)

        # Survival
        survival = 1.

        # Reach end platform
        reach_platform_end = (self.distance_to_platform_end - new_distance_to_platform_end) / self.dt

        return {"survival": survival, "reach_platform_end": reach_platform_end}

    def get_terminated(self, action):
        # Terminate if fall off
        return self.robot.links["base"].position[2] < self.platform_start_position[2]

    def get_success(self):
        # Success if reach end platform
        is_on_platform_end = self.get_distance_to_object(self.platform_end_id) < self.platform_size[2] / 2
        return is_on_platform_end
