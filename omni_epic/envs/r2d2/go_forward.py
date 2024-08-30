import numpy as np
from omni_epic.envs.r2d2.base import R2D2Env


class Env(R2D2Env):
    """
    Go forward.

    Description:
    The robot is standing on a flat ground represented by a box.
    The task of the robot is to go forward as fast as possible.

    Success:
    The task is completed if the robot runs forward for 10 meters.

    Rewards:
    The help the robot complete the task:
    - The robot is rewarded for survival at each time step.
    - The robot is rewarded for forward velocity, incentivizing it to move forward quickly.

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

        # Survival
        survival = 1.

        # Forward velocity
        forward_velocity = (new_position[0] - self.position[0]) / self.dt

        return {"survival": survival, "forward_velocity": forward_velocity}

    def get_terminated(self, action):
        # No termination
        return False

    def get_success(self):
        # Success if run forward for 10 meters
        return self.robot.links["base"].position[0] > 10.
