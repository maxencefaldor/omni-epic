import numpy as np
from omni_epic.envs.r2d2.base import R2D2Env


class Env(R2D2Env):
    """
    Cross over lava on a boat to reach a target zone.

    Description:
    - The lava is simulated with an orange, 10 x 10 m heightfield.
    - There are two platforms on either side of the lava, each measuring 5 x 10 m. One serves as the start platform and the other as the end platform.
    - The boat is a box with dimensions 3 meters in length, 2 meters in width, and 0.2 meters in height. It is initialized next to the start platform at a random y-position.
    - The boat has a button that, when pressed, activates the boat to move over the lava at a speed of 3 meters per second.
    - The end platform has a target zone indicated by a green, transparent sphere.
    The robot's task is to jump onto the boat from the start platform, press the button to activate the boat, and travel across the lava to reach the end platform. The robot must then enter the target zone to complete the task.

    Success:
    The task is successfully completed when the robot enters the target zone on the end platform.

    Rewards:
    To guide the robot to complete the task:
    - The robot receives a reward for each time step it remains active and does not fall off or touch the lava.
    - The robot is rewarded for making progress towards pressing the button on the boat.
    - Additional rewards are given for progressing towards the target zone, with a significant bonus for entering the target zone.

    Termination:
    The task terminates immediately if the robot falls off the platform or the boat, or if it touches the simulated lava.
    """

    def __init__(self):
        super().__init__()

        # Init lava
        self.lava_size = [10., 10.]
        self.lava_height = 0.1
        self.lava_position = [0., 0., 0.]
        self.lava_id = self.create_heightfield(
            size=self.lava_size,
            height_max=self.lava_height,  # create small bumps to create a fluid-like surface
            position=self.lava_position,
            resolution=20,  # number of points per meter
            repeats=2,
        )
        self._p.changeVisualShape(objectUniqueId=self.lava_id, linkIndex=-1, rgbaColor=[1., 0.3, 0.1, 1.])  # change to lava color

        # Init platforms
        self.platform_size = [5., self.lava_size[1], 1.]
        self.platform_start_position = [self.lava_position[0] - self.lava_size[0] / 2 - self.platform_size[0] / 2, self.lava_position[1], self.lava_position[2]]
        self.platform_end_position = [self.lava_position[0] + self.lava_size[0] / 2 + self.platform_size[0] / 2, self.lava_position[1], self.lava_position[2]]
        self.platform_start_id = self.create_box(mass=0., half_extents=[self.platform_size[0] / 2, self.platform_size[1] / 2, self.platform_size[2] / 2], position=self.platform_start_position, color=[0.3, 0.3, 0.3, 1.])
        self.platform_end_id = self.create_box(mass=0., half_extents=[self.platform_size[0] / 2, self.platform_size[1] / 2, self.platform_size[2] / 2], position=self.platform_end_position, color=[0.3, 0.3, 0.3, 1.])
        self._p.changeDynamics(bodyUniqueId=self.platform_start_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)
        self._p.changeDynamics(bodyUniqueId=self.platform_end_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

        # Init boat
        self.boat_size = [3., 2., 0.2]
        self.boat_position_init = [self.lava_position[0] - self.lava_size[0] / 2 + self.boat_size[0] / 2, self.lava_position[1], self.boat_size[2] / 2]
        self.boat_speed = 3.
        self.boat_id = self.create_box(mass=0., half_extents=[self.boat_size[0] / 2, self.boat_size[1] / 2, self.boat_size[2] / 2], position=self.boat_position_init, color=[0.8, 0.8, 0.8, 1.])
        self._p.changeDynamics(bodyUniqueId=self.boat_id, linkIndex=-1, lateralFriction=0.8, restitution=0.5)

        # Init button
        self.button_radius = 0.25
        self.button_height = 0.25
        self.button_position_init = [self.boat_position_init[0] + self.boat_size[0] / 4, self.lava_position[1], self.boat_position_init[2] + self.boat_size[2] / 2 + self.button_height / 2]  # put button on the right side of the boat
        self.button_id = self.create_cylinder(mass=0., radius=self.button_radius, height=self.button_height, position=self.button_position_init, color=[0., 0.5, 0., 1.])

        # Init target zone
        self.target_zone_radius = 1.5
        self.target_zone_id = self.create_sphere(mass=0., radius=self.target_zone_radius, collision=False, position=[self.platform_end_position[0], self.platform_end_position[1], self.platform_end_position[2] + self.platform_size[2] / 2], color=[0., 1., 0., 0.5])

        self.objects_on_boat = [self.button_id]

    def create_box(self, mass, half_extents, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def create_cylinder(self, mass, radius, height, position, color):
        collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_CYLINDER, radius=radius, height=height)
        visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def create_sphere(self, mass, radius, collision, position, color):
        if collision:
            collision_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_SPHERE, radius=radius)
            visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            return self._p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position)
        else:
            visual_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            return self._p.createMultiBody(baseMass=mass, baseVisualShapeIndex=visual_shape_id, basePosition=position)

    def create_heightfield(self, size, height_max, position, resolution, repeats=2):
        heightfield_data = np.random.uniform(low=0., high=height_max, size=(int(size[0] * resolution / repeats), int(size[1] * resolution / repeats)))
        heightfield_data = np.repeat(np.repeat(heightfield_data, repeats, axis=0), repeats, axis=1)
        mesh_scale = [1/resolution, 1/resolution, 1.]
        heightfield_collision_shape_id = self._p.createCollisionShape(
            shapeType=self._p.GEOM_HEIGHTFIELD,
            meshScale=mesh_scale,
            heightfieldData=heightfield_data.reshape(-1),
            numHeightfieldRows=heightfield_data.shape[0],
            numHeightfieldColumns=heightfield_data.shape[1],
        )
        return self._p.createMultiBody(baseMass=0., baseCollisionShapeIndex=heightfield_collision_shape_id, basePosition=[position[0], position[1], position[2] + mesh_scale[2] * height_max / 2])

    def get_object_position(self, object_id):
        return np.asarray(self._p.getBasePositionAndOrientation(object_id)[0])

    def get_distance_to_object(self, object_id):
        object_position = self.get_object_position(object_id)
        robot_position = self.robot.links["base"].position
        return np.linalg.norm(object_position[:2] - robot_position[:2])

    def reset(self):
        observation = super().reset()

        # Reset boat position
        boat_y_init = np.random.uniform(low=-self.lava_size[1] / 2 + self.boat_size[1] / 2, high=self.lava_size[1] / 2 - self.boat_size[1] / 2)  # randomize y position
        self._p.resetBasePositionAndOrientation(self.boat_id, [self.boat_position_init[0], boat_y_init, self.boat_position_init[2]], [0., 0., 0., 1.])

        # Reset button position
        self._p.resetBasePositionAndOrientation(self.button_id, [self.button_position_init[0], boat_y_init, self.button_position_init[2]], [0., 0., 0., 1.])

        # Reset target zone
        target_zone_y = np.random.uniform(low=-self.lava_size[1] / 2 + self.target_zone_radius, high=self.lava_size[1] / 2 - self.target_zone_radius)  # randomize y position
        self.target_zone_position = [self.platform_end_position[0], target_zone_y, self.platform_end_position[2] + self.platform_size[2] / 2]
        self._p.resetBasePositionAndOrientation(self.target_zone_id, self.target_zone_position, [0., 0., 0., 1.])

        # Reset robot position
        self._p.resetBasePositionAndOrientation(self.robot.robot_id, [self.platform_start_position[0], self.platform_start_position[1], self.platform_start_position[2] + self.platform_size[2] / 2 + self.robot.links["base"].position_init[2]], self.robot.links["base"].orientation_init)

        return observation

    def step(self, action):
        # Before taking action
        self.distance_to_button = self.get_distance_to_object(self.button_id)
        self.distance_to_target_zone = self.get_distance_to_object(self.target_zone_id)
        self.has_touched_platform_end = len(self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.platform_end_id)) > 0

        observation, reward, terminated, truncated, info = super().step(action)

        # Check if button is pressed
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.button_id)
        button_pressed = len(contact_points) > 0

        if button_pressed:
            # Move boat and everything on boat forward
            for body_id in [self.boat_id] + self.objects_on_boat:
                body_position = self.get_object_position(body_id)
                new_object_position = body_position + np.array([self.boat_speed * self.dt, 0., 0.])
                self._p.resetBasePositionAndOrientation(body_id, new_object_position, [0., 0., 0., 1.])

        return observation, reward, terminated, truncated, info

    def get_task_rewards(self, action):
        # After taking action
        new_distance_to_button = self.get_distance_to_object(self.button_id)
        new_distance_to_target_zone = self.get_distance_to_object(self.target_zone_id)

        # Survival
        survival = 1.

        # Reach button
        reach_button = (self.distance_to_button - new_distance_to_button) / self.dt

        # Reach target zone
        reach_target_zone = (self.distance_to_target_zone - new_distance_to_target_zone) / self.dt
        if self.distance_to_target_zone < self.target_zone_radius:
            reach_target_zone += 5.

        return {"survival": survival, "reach_button": reach_button, "reach_target_zone": reach_target_zone}

    def get_terminated(self, action):
        # Terminate if touch lava
        contact_points = self._p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.lava_id)
        is_touching_lava = len(contact_points) > 0

        # Terminate if fall off
        is_fall_off = self.robot.links["base"].position[2] < self.platform_start_position[2]
        return is_touching_lava or is_fall_off

    def get_success(self):
        # Success if stand in the target zone
        distance_to_target_zone = self.get_distance_to_object(self.target_zone_id)
        return distance_to_target_zone < self.target_zone_radius
