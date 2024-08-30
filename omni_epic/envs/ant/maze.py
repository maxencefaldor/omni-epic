import numpy as np
from omni_epic.envs.ant.base import AntEnv


class Env(AntEnv):
    """
    Navigate through a maze to reach the end position.

    Description:
    - The environment consists of a large flat ground measuring 1000 x 1000 x 10 m.
    - A maze is constructed on the ground using walls of height 1 meter and scale 3 m per cell.
    - The maze is represented by a 2D array where 0 indicates an empty space, 1 indicates a wall, 2 indicates the start position, and 3 indicates the end position.
    - The robot is initialized at the start position in the maze.
    - The task of the robot is to navigate through the maze and reach the end position.

    Success:
    The task is completed if the robot reaches the end position in the maze.

    Rewards:
    To guide the robot to complete the task:
    - The robot receives a reward at each time step for survival.
    - The robot is rewarded for making progress towards the end position in the maze.

    Termination:
    The task terminates if the robot falls.
    """

    def __init__(self):
        super().__init__()

        # Init ground
        self.ground_size = [1000., 1000., 10.]
        self.ground_position = [0., 0., 0.]
        self.ground_id = self.create_box(mass=0., half_extents=[self.ground_size[0] / 2, self.ground_size[1] / 2, self.ground_size[2] / 2], position=self.ground_position, color=[0.5, 0.5, 0.5, 1.])
        self._p.changeDynamics(bodyUniqueId=self.ground_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

        # Init maze - 0 is empty, 1 is wall, 2 is start, 3 is end
        self.maze_height = 1.
        self.maze_scale = 3.
        maze = np.array([
            [1, 1, 1, 3, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1],
            [2, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ])
        for index, value in np.ndenumerate(maze):
                if value == 1:
                    self.create_box(0., half_extents=[self.maze_scale / 2, self.maze_scale / 2, self.maze_height / 2], position=[self.maze_scale * index[1], -self.maze_scale * index[0], self.ground_position[2] + self.ground_size[2] / 2 + self.maze_height / 2], color=[0.2, 0.2, 0.2, 1])

        # Get start and end position
        start_position_index = np.squeeze(np.argwhere(maze == 2))
        self.start_position = self.maze_scale * np.array([start_position_index[1], -start_position_index[0]])
        end_position_index = np.squeeze(np.argwhere(maze == 3))
        self.end_position = self.maze_scale * np.array([end_position_index[1], -end_position_index[0]])

    def create_box(self, mass, half_extents, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def reset(self):
        observation = super().reset()

        # Reset robot position at start position
        self._p.resetBasePositionAndOrientation(self.robot.robot_id, [self.start_position[0], self.start_position[1], self.ground_position[2] + self.ground_size[2] / 2 + self.robot.links["base"].position_init[2]], self.robot.links["base"].orientation_init)

        return observation

    def step(self, action):
        # Before taking action
        self.position = self.robot.links["base"].position
        self.distance_to_end = np.linalg.norm(self.position[:2] - self.end_position)

        observation, reward, terminated, truncated, info = super().step(action)

        return observation, reward, terminated, truncated, info

    def get_task_rewards(self, action):
        # After taking action
        new_position = self.robot.links["base"].position
        new_distance_to_end = np.linalg.norm(new_position[:2] - self.end_position)

        # Standing
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, linkIndexA=self.robot.links["base"].index)
        objects_in_contact = {contact_point[2] for contact_point in contact_points}
        standing = 1. if self.ground_id not in objects_in_contact else -1.

        # Progress in the maze
        maze_progress = (self.distance_to_end - new_distance_to_end) / self.dt

        return {"standing": standing, "maze_progress": maze_progress}

    def get_terminated(self, action):
        # Terminate if not standing
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, linkIndexA=self.robot.links["base"].index)
        objects_in_contact = {contact_point[2] for contact_point in contact_points}
        is_standing = self.ground_id not in objects_in_contact  # allow body to touch walls
        return not is_standing

    def get_success(self):
        # Success if reach end of maze
        return np.linalg.norm(self.robot.links["base"].position[:2] - self.end_position) < self.maze_scale
