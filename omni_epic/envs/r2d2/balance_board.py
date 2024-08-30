import numpy as np
from omni_epic.envs.r2d2.base import R2D2Env


class Env(R2D2Env):
    """
    Balance on a board placed on top of a rolling cylinder.

    Description:
    - A cylinder (radius 0.5 m and height 1 m) is placed on the ground and can roll along the y-axis.
    - A board (length 3 m, width 2 m and thickness 0.05 m) is placed on top of the cylinder.
    The robot is initialized on top of the board facing toward the positive x-axis.
    The task of the robot is to stand on the board and keep its balance on the board while the cylinder moves underneath.

    Success:
    The task is completed if the robot remains standing on the board for more than 10 s.

    Rewards:
    The robot is rewarded for remaining on the board.

    Termination:
    The task terminates if the robot falls of the board.
    """

    def __init__(self):
        super().__init__()

        # Init ground
        self.ground_size = [10., 10., 10.]
        self.ground_position = [0., 0., 0.]
        self.ground_id = self.create_box(mass=0., half_extents=[self.ground_size[0] / 2, self.ground_size[1] / 2, self.ground_size[2] / 2], position=self.ground_position, color=[0.5, 0.5, 0.5, 1.])
        self._p.changeDynamics(bodyUniqueId=self.ground_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

        # Init cylinder
        self.cylinder_radius = 0.5
        self.cylinder_height = 1.
        self.cylinder_position_init = [self.ground_position[0], self.ground_position[1], self.ground_position[2] + self.ground_size[2] / 2 + self.cylinder_radius]
        self.cylinder_orientation_init = self._p.getQuaternionFromEuler(eulerAngles=[0., np.pi / 2, 0.])  # roll along y-axis
        self.cylinder_id = self.create_cylinder(mass=10., radius=self.cylinder_radius, height=self.cylinder_height, position=self.cylinder_position_init, orientation=self.cylinder_orientation_init, color=[0., 0., 1., 1.]) 
        self._p.changeDynamics(bodyUniqueId=self.cylinder_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

        # Init board
        self.board_size = [2., 3., 0.05]
        self.board_position_init = [self.cylinder_position_init[0], self.cylinder_position_init[1], self.cylinder_position_init[2] + self.cylinder_radius + self.board_size[2] / 2]  # Init board above cylinder
        self.board_id = self.create_box(mass=10., half_extents=[self.board_size[0] / 2, self.board_size[1] / 2, self.board_size[2] / 2], position=self.board_position_init, color=[1., 0., 0., 1.])
        self._p.changeDynamics(bodyUniqueId=self.board_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

    def create_box(self, mass, half_extents, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def create_cylinder(self, mass, radius, height, position, orientation, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_CYLINDER, radius=radius, height=height)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=orientation)

    def reset(self):
        observation = super().reset()

        # Reset time
        self.time = 0.

        # Reset cylinder position
        self._p.resetBasePositionAndOrientation(self.cylinder_id, self.cylinder_position_init, self.cylinder_orientation_init)

        # Reset board position
        self._p.resetBasePositionAndOrientation(self.board_id, self.board_position_init, [0., 0., 0., 1.])

        # Reset robot position
        self._p.resetBasePositionAndOrientation(self.robot.robot_id, [self.board_position_init[0], self.board_position_init[1], self.board_position_init[2] + self.board_size[2] / 2 + self.robot.links["base"].position_init[2]], self.robot.links["base"].orientation_init)

        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        # Increase time
        self.time += self.dt

        return observation, reward, terminated, truncated, info

    def get_task_rewards(self, action):
        # On board
        on_board = 1. if self.robot.links["base"].position[2] > self.board_position_init[2] + self.board_size[2] / 2 else -1.
        return {"on_board": on_board}

    def get_terminated(self, action):
        # Terminate if not on board
        is_on_board = self.robot.links["base"].position[2] > self.board_position_init[2] + self.board_size[2] / 2
        return not is_on_board

    def get_success(self):
        # Success if on board after 10. s
        on_board = self.time >= 10. and self.robot.links["base"].position[2] > self.board_position_init[2] + self.board_size[2] / 2
        return on_board
