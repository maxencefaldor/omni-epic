import numpy as np
from omni_epic.envs.r2d2.base import R2D2Env


class Env(R2D2Env):
    """
    Reach a box.

    Description:
    - The environment consists of a large flat ground measuring 1000 x 1000 x 10 meters.
    - A box with dimensions 1 x 1 x 1 meter is placed randomly on the ground in a radius of 25 m around the robot. To avoid collisions, the box cannot spawn in a radius of 2 m around the robot.
    - The robot is initialized at a fixed position on the ground.
    The task of the robot is to reach and touch the box.

    Success:
    The task is completed if the robot makes contact with the box.

    Rewards:
    To help the robot complete the task:
    - The robot is rewarded for survival.
    - The robot is rewarded for moving closer to the box.

    Termination:
    None.
    """

    def __init__(self):
        super().__init__()

        # Init ground
        self.ground_size = [1000., 1000., 10.]
        self.ground_position = [0., 0., 0.]
        self.ground_id = self.create_box(mass=0., half_extents=[self.ground_size[0] / 2, self.ground_size[1] / 2, self.ground_size[2] / 2], position=self.ground_position, color=[0.5, 0.5, 0.5, 1.])
        self._p.changeDynamics(bodyUniqueId=self.ground_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

        # Init box
        self.box_size = [1., 1., 1.]
        self.box_id = self.create_box(mass=1., half_extents=[self.box_size[0] / 2, self.box_size[1] / 2, self.box_size[2] / 2], position=[0., 0., 0.], color=[1., 0., 0., 1.])

        # Starting position of the robot
        self.robot_position_init = [self.ground_position[0], self.ground_position[1], self.ground_position[2] + self.ground_size[2] / 2 + self.robot.links["base"].position_init[2]]

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

        # Reset robot position on ground
        self._p.resetBasePositionAndOrientation(self.robot.robot_id, self.robot_position_init, self.robot.links["base"].orientation_init)

        # Reset box position
        angle = np.random.uniform(0., 2 * np.pi)
        radius = np.random.uniform(2., 25.)
        self._p.resetBasePositionAndOrientation(self.box_id, [self.robot_position_init[0] + radius * np.cos(angle), self.robot_position_init[1] + radius * np.sin(angle), self.ground_position[2] + self.ground_size[2] / 2 + self.box_size[2] / 2], [0., 0., 0., 1.])

        return observation

    def step(self, action):
        # Before taking action
        self.distance_to_box = self.get_distance_to_object(self.box_id)

        observation, reward, terminated, truncated, info = super().step(action)

        return observation, reward, terminated, truncated, info

    def get_task_rewards(self, action):
        # After taking action
        new_distance_to_box = self.get_distance_to_object(self.box_id)

        # Survival
        survival = 1.

        # Reach box
        reach_box = (self.distance_to_box - new_distance_to_box) / self.dt

        return {"survival": survival, "reach_box": reach_box}

    def get_terminated(self, action):
        # No termination
        return False

    def get_success(self):
        # Success if touch box
        contact_points_box = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.box_id)
        is_touching_box = len(contact_points_box) > 0
        return is_touching_box
