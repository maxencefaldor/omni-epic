import numpy as np
from omni_epic.envs.humanoid.base import HumanoidEnv


class Env(HumanoidEnv):
    """
    Go forward on top of a rolling cylinder.

    Description:
    - The environment consists of a large flat ground measuring 1000 x 1000 x 10 m.
    - A cylinder with a radius of 2 m and a height of 3 m is placed on the ground and can roll along the x-axis.
    - The cylinder's initial position is at the center of the ground, and it is oriented to roll along the x-axis.
    - The robot is initialized on top of the cylinder.
    - The task of the robot is to go forward while balancing on top of the rolling cylinder.

    Success:
    The task is completed if the robot rolls more than 5 m forward without falling off.

    Rewards:
    To guide the robot to complete the task:
    - The robot receives a reward for each time step it remains balanced on the cylinder.
    - The robot receives a reward for forward velocity along the x-axis.

    Termination:
    The task terminates immediately if the is not standing on the cylinder or if the robot falls off the cylinder.
    """

    def __init__(self):
        super().__init__()

        # Init ground
        self.ground_size = [1000., 1000., 10.]
        self.ground_position = [0., 0., 0.]
        self.ground_id = self.create_box(mass=0., half_extents=[self.ground_size[0] / 2, self.ground_size[1] / 2, self.ground_size[2] / 2], position=self.ground_position, color=[0.5, 0.5, 0.5, 1.])
        self._p.changeDynamics(bodyUniqueId=self.ground_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

        # Init cylinder
        self.cylinder_radius = 2.
        self.cylinder_height = 3.
        self.cylinder_position_init = [self.ground_position[0], self.ground_position[1], self.ground_position[2] + self.ground_size[2] / 2 + self.cylinder_radius]
        self.cylinder_orientation_init = self._p.getQuaternionFromEuler(eulerAngles=[np.pi / 2, 0., 0.])  # roll along x-axis
        self.cylinder_id = self.create_cylinder(mass=25., radius=self.cylinder_radius, height=self.cylinder_height, position=self.cylinder_position_init, orientation=self.cylinder_orientation_init, color=[0., 0., 1., 1.]) 
        self._p.changeDynamics(bodyUniqueId=self.cylinder_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

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

        # Reset cylinder position
        self._p.resetBasePositionAndOrientation(self.cylinder_id, self.cylinder_position_init, self.cylinder_orientation_init)

        # Reset robot position on the top of cylinder
        self._p.resetBasePositionAndOrientation(self.robot.robot_id, [self.cylinder_position_init[0], self.cylinder_position_init[1], self.cylinder_position_init[2] + self.cylinder_radius + self.robot.links["base"].position_init[2]], self.robot.links["base"].orientation_init)

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
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.cylinder_id)
        objects_in_contact = {contact_point[2] for contact_point in contact_points if contact_point[3] not in {self.robot.links["left_foot"].index, self.robot.links["right_foot"].index}}
        standing = 2. if len(objects_in_contact) == 0 else -1.

        # Forward velocity
        forward_velocity = (new_position[0] - self.position[0]) / self.dt

        return {"standing": standing, "forward_velocity": forward_velocity}

    def get_terminated(self, action):
        # Terminate if not standing
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.cylinder_id)
        objects_in_contact = {contact_point[2] for contact_point in contact_points if contact_point[3] not in {self.robot.links["left_foot"].index, self.robot.links["right_foot"].index}}
        is_standing = len(objects_in_contact) == 0

        # Terminate if not on cylinder or not standing
        is_on_cylinder = self.robot.links["base"].position[2] > 2 * self.cylinder_radius

        return not is_standing or not is_on_cylinder

    def get_success(self):
        # Success if rolled on cylinder
        return self.robot.links["base"].position[0] > 5.
