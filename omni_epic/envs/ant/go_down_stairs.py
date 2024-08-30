import numpy as np
from omni_epic.envs.ant.base import AntEnv


class Env(AntEnv):
    """
    Descend a series of stairs to reach the ground.

    Description:
    - The environment consists of a ground platform (1000 m x 10 m x 10 m) and a set of 10 steps.
    - Each step has dimensions of 1 m in length, 10 m in width, and 0.2 m in height.
    - The steps are positioned to form a descending staircase starting from an initial height, with each subsequent step lower than the previous one.
    The robot is initialized at the top of the stairs.

    Success:
    The task is completed when the robot successfully descends the stairs and touches the ground platform.

    Rewards:
    The help the robot complete the task:
    - The robot is rewarded for standing at each time step.
    - The robot is rewarded for forward velocity, incentivizing it to move down the stairs.

    Termination:
    The task terminates immediately if the robot falls off the stairs or the ground platform.
    """

    def __init__(self):
        super().__init__()

        # Init ground
        self.ground_size = [1000., 10., 10.]
        self.ground_position = [0., 0., 0.]
        self.ground_id = self.create_box(mass=0., half_extents=[self.ground_size[0] / 2, self.ground_size[1] / 2, self.ground_size[2] / 2], position=self.ground_position, color=[0.5, 0.5, 0.5, 1.])
        self._p.changeDynamics(bodyUniqueId=self.ground_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

        # Init stairs
        self.num_steps = 10
        self.step_size = [1.0, 10., 0.2]
        self.step_position_init = [self.ground_position[0], self.ground_position[1], self.ground_position[2] + self.ground_size[2] / 2 + self.num_steps * self.step_size[2]]
        self.create_stairs_down(step_size=self.step_size, step_position_init=self.step_position_init, num_steps=self.num_steps)

    def create_box(self, mass, half_extents, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def create_stairs_down(self, step_size, step_position_init, num_steps):
        color_1 = np.array([1., 0., 0.])
        color_2 = np.array([0., 0., 1.])
        for i in range(num_steps):
            step_position = [step_position_init[0] + i * step_size[0], step_position_init[1], step_position_init[2] - i * step_size[2]]
            interpolation = i / (num_steps - 1)
            step_color = (1 - interpolation) * color_1 + interpolation * color_2  # shade steps for visualization
            self.create_box(mass=0., half_extents=[step_size[0] / 2, step_size[1] / 2, step_size[2] / 2], position=step_position, color=np.append(step_color, 1.))

    def reset(self):
        observation = super().reset()

        # Reset robot position at the top of the stairs
        self._p.resetBasePositionAndOrientation(self.robot.robot_id, [self.step_position_init[0], self.step_position_init[1], self.step_position_init[2] + self.step_size[2] / 2 + self.robot.links["base"].position_init[2]], self.robot.links["base"].orientation_init)

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
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, linkIndexA=self.robot.links["base"].index)
        standing = 1. if len(contact_points) == 0 else -1.

        # Forward velocity
        forward_velocity = (new_position[0] - self.position[0]) / self.dt

        return {"standing": standing, "forward_velocity": forward_velocity}

    def get_terminated(self, action):
        # Terminate if not standing
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, linkIndexA=self.robot.links["base"].index)
        is_standing = len(contact_points) == 0

        # Terminate if fall off
        is_fall_off = self.robot.links["base"].position[2] < self.ground_position[2]
        return not is_standing or is_fall_off

    def get_success(self):
        # Success if reach end stairs and touch ground
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.ground_id)
        is_on_ground = len(contact_points) > 0
        return is_on_ground
