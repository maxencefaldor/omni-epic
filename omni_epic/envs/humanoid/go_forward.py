import numpy as np
from omni_epic.envs.humanoid.base import HumanoidEnv


class Env(HumanoidEnv):
    """
    Go forward.

    Description:
    The robot is standing on a flat ground represented by a box.
    The task of the robot is to go forward as fast as possible.

    Success:
    The task is completed if the robot runs forward for 10 meters.

    Rewards:
    The help the robot complete the task:
    - The robot is rewarded for standing at each time step.
    - The robot is rewarded for forward velocity, incentivizing it to move forward quickly.

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

    def create_box(self, mass, half_extents, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def reset(self):
        observation = super().reset()

        # Reset robot position
        self._p.resetBasePositionAndOrientation(self.robot.robot_id, [self.ground_position[0], self.ground_position[1], self.ground_position[2] + self.ground_size[2] / 2 + self.robot.links["base"].position_init[2]], self.robot.links["base"].orientation_init)

        return observation

    def step(self, action):
        # Before taking action
        self.position = self.robot.links["base"].position

        observation, reward, terminated, truncated, info = super().step(action)

        return observation, reward, terminated, truncated, info

    def get_task_rewards(self, action):
        # After taking action
        new_position = self.robot.links["base"].position

        # Standing
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.ground_id)
        objects_in_contact = {contact_point[2] for contact_point in contact_points if contact_point[3] not in {self.robot.links["left_foot"].index, self.robot.links["right_foot"].index}}
        standing = 2. if len(objects_in_contact) == 0 else -1.

        # Forward velocity
        forward_velocity = (new_position[0] - self.position[0]) / self.dt

        return {"standing": standing, "forward_velocity": forward_velocity}

    def get_terminated(self, action):
        # Terminate if not standing
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.ground_id)
        objects_in_contact = {contact_point[2] for contact_point in contact_points if contact_point[3] not in {self.robot.links["left_foot"].index, self.robot.links["right_foot"].index}}
        is_standing = len(objects_in_contact) == 0
        return not is_standing

    def get_success(self):
        # Success if run forward for 10 meters
        return self.robot.links["base"].position[0] > 10.
