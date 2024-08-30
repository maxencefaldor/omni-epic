import numpy as np
from omni_epic.envs.humanoid.base import HumanoidEnv


class Env(HumanoidEnv):
    """
    Activate a lever to open a door and move through the door.

    Description:
    - The environment consists of a large platform measuring 1000 x 10 x 0.1 meters.
    - The robot is initialized at a fixed position on the platform.
    - A door with dimensions 0.5 x 2 x 2 meters is placed on the platform, 5 m away from the robot, initially closed.
    - The door is flanked by walls to prevent the robot from bypassing it.
    - A lever is placed on the platform, 2 meters to the left of the door.
    - The task of the robot is to move to the lever, activate it to open the door, and then pass through the door.

    Success:
    The task is successfully completed if the robot passes through the door and moves more than 10 m beyond the initial position.

    Rewards:
    To guide the robot to complete the task:
    - The robot receives a survival reward at each time step.
    - The robot is rewarded for decreasing its distance to the lever.
    - The robot receives a bonus rewards for activating the lever to open the door.
    - Once the door is open, the robot is rewarded for moving forward.

    Termination:
    The task terminates immediately if the robot falls off the stairs or the ground platform.
    """

    def __init__(self):
        super().__init__()

        self.robot_position_init = [0., 0., 0.]

        # Init platform
        self.platform_size = [1000., 10., 0.1]
        self.platform_position = [self.robot_position_init[0] + self.platform_size[0] / 2 - 2., self.robot_position_init[1], self.robot_position_init[2] - self.platform_size[2] / 2]  # offset by 2 m to avoid off-edge or on-edge placement
        self.platform_id = self.create_box(mass=0., half_extents=[self.platform_size[0] / 2, self.platform_size[1] / 2, self.platform_size[2] / 2], position=self.platform_position, color=[0.5, 0.5, 0.5, 1.])
        self._p.changeDynamics(bodyUniqueId=self.platform_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

        # Init door
        self.door_size = [0.5, 2., 2.]
        self.door_position_init = [self.robot_position_init[0] + 5., self.platform_position[1], self.platform_position[2] + self.platform_size[2] / 2 + self.door_size[2] / 2]
        self.door_id = self.create_box(mass=0., half_extents=[self.door_size[0] / 2, self.door_size[1] / 2, self.door_size[2] / 2], position=self.door_position_init, color=[1., 0., 0., 1.])
        self.door_open = False

        # Init wall
        self.wall_size = [self.door_size[0], (self.platform_size[1] - self.door_size[1]) / 2, self.door_size[2]]  # walls plus door span the full platform to prevent robot to go around
        self.create_box(mass=0., half_extents=[self.wall_size[0] / 2, self.wall_size[1] / 2, self.wall_size[2] / 2], position=[self.door_position_init[0], self.door_position_init[1] + self.door_size[1] / 2 + self.wall_size[1] / 2, self.platform_position[2] + self.platform_size[2] / 2 + self.wall_size[2] / 2], color=[0., 0., 1., 1.])  # left section
        self.create_box(mass=0., half_extents=[self.wall_size[0] / 2, self.wall_size[1] / 2, self.wall_size[2] / 2], position=[self.door_position_init[0], self.door_position_init[1] - self.door_size[1] / 2 - self.wall_size[1] / 2, self.platform_position[2] + self.platform_size[2] / 2 + self.wall_size[2] / 2], color=[0., 0., 1., 1.])  # right section

        # Init lever
        self.lever_radius = 0.05
        self.lever_height = 0.5
        lever_position = [self.door_position_init[0] - 2., self.door_size[1], self.platform_position[2] + self.platform_size[2] / 2 + self.lever_height / 2]  # two meters to the left of the door on the platform
        self.lever_id = self.create_cylinder(mass=0., radius=self.lever_radius, height=self.lever_height, position=lever_position, color=[0.5, 0.25, 0., 1.])

    def create_box(self, mass, half_extents, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def create_cylinder(self, mass, radius, height, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_CYLINDER, radius=radius, height=height)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def get_object_position(self, object_id):
        return np.asarray(self._p.getBasePositionAndOrientation(object_id)[0])

    def get_distance_to_object(self, object_id):
        object_position = self.get_object_position(object_id)
        robot_position = self.robot.links["base"].position
        return np.linalg.norm(object_position[:2] - robot_position[:2])

    def reset(self):
        observation = super().reset()

        # Reset door
        self.door_open = False
        self._p.resetBasePositionAndOrientation(self.door_id, self.door_position_init, [0., 0., 0., 1.])

        # Reset robot position
        self._p.resetBasePositionAndOrientation(self.robot.robot_id, [self.robot_position_init[0], self.robot_position_init[1], self.robot_position_init[2] + self.robot.links["base"].position_init[2]], self.robot.links["base"].orientation_init)

        return observation

    def step(self, action):
        # Before taking action
        self.position = self.robot.links["base"].position
        self.distance_to_lever = self.get_distance_to_object(self.lever_id)

        observation, reward, terminated, truncated, info = super().step(action)

        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.lever_id)
        if len(contact_points) > 0 and not self.door_open:
            self.door_open = True
            self._p.resetBasePositionAndOrientation(self.door_id, [self.door_position_init[0], self.door_position_init[1] + self.door_size[1], self.door_position_init[2]], [0., 0., 0., 1.])

        return observation, reward, terminated, truncated, info

    def get_task_rewards(self, action):
        # After taking action
        new_position = self.robot.links["base"].position
        new_distance_to_lever = self.get_distance_to_object(self.lever_id)

        # Standing
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.platform_id)
        objects_in_contact = {contact_point[2] for contact_point in contact_points if contact_point[3] not in {self.robot.links["left_foot"].index, self.robot.links["right_foot"].index}}
        standing = 2. if len(objects_in_contact) == 0 else -1.

        # Reach lever
        if not self.door_open and len(self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.lever_id)) == 0:
            reach_lever = (self.distance_to_lever - new_distance_to_lever) / self.dt
        elif not self.door_open and len(self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.lever_id)) > 0:
            reach_lever = 10.
        else:
            reach_lever = 0.

        # Forward velocity
        if self.door_open:
            forward_velocity = (new_position[0] - self.position[0]) / self.dt
        else:
            forward_velocity = 0.

        return {"standing": standing, "reach_lever": reach_lever, "forward_velocity": forward_velocity}

    def get_terminated(self, action):
        # Terminate if not standing
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.platform_id)
        objects_in_contact = {contact_point[2] for contact_point in contact_points if contact_point[3] not in {self.robot.links["left_foot"].index, self.robot.links["right_foot"].index}}
        is_standing = len(objects_in_contact) == 0

        # Terminate if fall off
        is_fall_off = self.robot.links["base"].position[2] < self.platform_position[2]
        return not is_standing or is_fall_off

    def get_success(self):
        # Success if pass through door
        return self.robot.links["base"].position[0] > 10.
