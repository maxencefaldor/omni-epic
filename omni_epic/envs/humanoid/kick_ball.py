import numpy as np
from omni_epic.envs.humanoid.base import HumanoidEnv


class Env(HumanoidEnv):
    """
    Kick a ball.

    Description:
    - The environment consists of a large flat ground measuring 1000 x 1000 x 10 meters.
    - A ball with a radius of 0.5 meters is placed randomly on the ground.
    - The robot is initialized at a fixed position on the ground.
    - The task of the robot is to move across the ground, reach the ball, and kick it as far away as possible.

    Success:
    The task is successfully completed if the robot kicks the ball so that it moves more than 10 meters away from its initial position.

    Rewards:
    To help the robot complete the task:
    - The robot is rewarded for standing.
    - The robot is rewarded for decreasing its distance to the ball.
    - The robot is rewarded for increasing the velocity of the ball to guide the robot to kick the ball.

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

        # Init ball
        self.ball_radius = 0.5
        self.ball_id = self.create_sphere(mass=1., radius=self.ball_radius, position=[0., 0., 0.], color=[1., 0., 0., 1.])

        # Starting position of the robot
        self.robot_position_init = [self.ground_position[0], self.ground_position[1], self.ground_position[2] + self.ground_size[2] / 2 + self.robot.links["base"].position_init[2]]

    def create_box(self, mass, half_extents, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def create_sphere(self, mass, radius, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_SPHERE, radius=radius)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_SPHERE, radius=radius, rgbaColor=color)
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

        # Reset ball position
        ball_y_init = np.random.uniform(self.robot_position_init[1] - 2., self.robot_position_init[1] + 2.)
        self._p.resetBasePositionAndOrientation(self.ball_id, [self.robot_position_init[0] + 5., ball_y_init, self.ground_position[2] + self.ground_size[2] / 2 + self.ball_radius], [0., 0., 0., 1.])

        return observation

    def step(self, action):
        # Before taking action
        self.distance_to_ball = self.get_distance_to_object(self.ball_id)
        self.ball_position = self.get_object_position(self.ball_id)

        observation, reward, terminated, truncated, info = super().step(action)

        return observation, reward, terminated, truncated, info

    def get_task_rewards(self, action):
        # After taking action
        new_distance_to_ball = self.get_distance_to_object(self.ball_id)
        new_ball_position = self.get_object_position(self.ball_id) 

        # Standing
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id)
        objects_in_contact = {contact_point[2] for contact_point in contact_points if contact_point[2] != self.robot.robot_id and contact_point[3] not in {self.robot.links["left_foot"].index, self.robot.links["right_foot"].index}}
        standing = 2. if self.ground_id not in objects_in_contact else -1.

        # Reach ball
        reach_ball = (self.distance_to_ball - new_distance_to_ball) / self.dt

        # Velocity of ball
        ball_velocity = np.linalg.norm(new_ball_position - self.ball_position) / self.dt

        return {"standing": standing, "reach_ball": reach_ball, "ball_velocity": ball_velocity}

    def get_terminated(self, action):
        # Terminate if not standing
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id)
        objects_in_contact = {contact_point[2] for contact_point in contact_points if contact_point[2] != self.robot.robot_id and contact_point[3] not in {self.robot.links["left_foot"].index, self.robot.links["right_foot"].index}}
        is_standing = self.ground_id not in objects_in_contact  # allow body to touch ball
        return not is_standing

    def get_success(self):
        # Success if kick ball 10 meters away from origin
        ball_distance_to_origin = np.linalg.norm(self.get_object_position(self.ball_id))
        return ball_distance_to_origin > 10.
