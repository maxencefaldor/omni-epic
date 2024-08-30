import importlib

import numpy as np
from scipy.spatial.transform import Rotation
import gym
import pybullet
from pybullet_utils import bullet_client
import pybullet_data

from omni_epic.robots.base import Robot, angle_between_vectors_2d


class Env(gym.Env):
	dt: float
	robot: Robot

	def __init__(self):
		self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

		# Load EGL renderer plugin
		self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
		egl = importlib.util.find_spec("eglRenderer")
		self._p.loadPlugin(egl.origin, "_eglRendererPlugin")

		# Init world
		self._init()

	def _init(self):
		# Set additional search path
		self._p.setAdditionalSearchPath("/workspace/src/omni_epic/envs/assets")

		# Reset simulation
		self._p.resetSimulation()

		# Set simulation parameters
		self._p.setGravity(0, 0, -9.8)
		self._p.setDefaultContactERP(0.9)
		self._p.setPhysicsEngineParameter(
			fixedTimeStep=self.dt,
			numSolverIterations=5,
			numSubSteps=4,
			deterministicOverlappingPairs=1,
		)

	def reset(self, seed=None, options=None):
		# Reset simulation
		super().reset(seed=seed)
		self.robot.reset(seed=seed)

		# Get observation
		return self.robot.get_observation()

	def step(self, action):
		# Step simulation
		self.robot.apply_action(action)
		self._p.stepSimulation()
		self.robot.update()

		# Get observation, reward, terminated and truncated
		observation = self.robot.get_observation()
		reward = self.get_reward(action)
		terminated = self.get_terminated(action)
		truncated = self.get_truncated(action)

		return observation, reward, terminated, truncated, {}

	def get_reward(self, action):
		task_rewards = self.get_task_rewards(action)
		robot_rewards = self.robot.get_rewards(action)
		return sum(task_rewards.values()) + sum(robot_rewards.values())

	def get_task_rewards(self, action):
		raise NotImplementedError

	def get_terminated(self, action):
		raise NotImplementedError

	def get_truncated(self, action):
		raise NotImplementedError

	def get_success(self):
		raise NotImplementedError

	def get_render_config(self):
		"""Get render config."""
		# Get AABB of robot
		aabb_min, aabb_max = self._p.getAABB(self.robot.robot_id)
		aabb_max = np.array(aabb_max)
		aabb_min = np.array(aabb_min)

		# Iterate over the rest of the bodies
		for body_id in range(self._p.getNumBodies()):
			# Skip robot id
			if body_id == self.robot.robot_id:
				continue
			try:
				# Get AABB of object
				obj_aabb_min, obj_aabb_max = self._p.getAABB(body_id)
				obj_aabb_max = np.array(obj_aabb_max)
				obj_aabb_min = np.array(obj_aabb_min)
				# If object is greater than a certain size, skip
				if np.linalg.norm(obj_aabb_max - obj_aabb_min) > 30.0:
					continue
				# Update AABB
				aabb_min = np.minimum(aabb_min, obj_aabb_min)
				aabb_max = np.maximum(aabb_max, obj_aabb_max)
			except:
				continue

		# Calculate render configs
		camera_target_position = (aabb_min + aabb_max) / 2
		distance = max([abs(x - y) for x, y in zip(camera_target_position, aabb_max)])
		return {
			"fov": 90,
			"camera_target_position": camera_target_position,
			"distance": distance + 0.5,
		}

	def render(self, height=720, width=1280, fov=60., camera_target_position=None, distance=10., yaw=0.):
		if camera_target_position is None:
			camera_target_position = self.robot.links["base"].position
		view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=camera_target_position,
			distance=distance,
			yaw=yaw,
			pitch=-30.,
			roll=0.,
			upAxisIndex=2,
		)
		proj_matrix = self._p.computeProjectionMatrixFOV(
			fov=fov,
			aspect=width / height,
			nearVal=0.01,
			farVal=100.0,
		)
		(_, _, rgba, _, _) = self._p.getCameraImage(
			width=width,
			height=height,
			viewMatrix=view_matrix,
			projectionMatrix=proj_matrix,
			renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
			flags=pybullet.ER_NO_SEGMENTATION_MASK,
		)
		return rgba[..., :3]

	def render3p(self, height=720, width=1280, fov=60., distance=5.):
		base_rotation_init = Rotation.from_quat(self.robot.links["base"].orientation_init)
		base_rotation = Rotation.from_quat(self.robot.links["base"].orientation)
		base_rotation_relative = base_rotation * base_rotation_init.inv()
		forward_vector = base_rotation_relative.apply(np.array([1., 0., 0.]))
		angle = angle_between_vectors_2d(forward_vector[:2], np.array([1., 0.]))
		view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=self.robot.links["base"].position + np.array([0., 0., 1.]),
			distance=distance,
			yaw=-np.degrees(angle) - 90.,
			pitch=-20.,
			roll=0.,
			upAxisIndex=2,
		)
		proj_matrix = self._p.computeProjectionMatrixFOV(
			fov=fov,
			aspect=width / height,
			nearVal=0.01,
			farVal=100.0,
		)
		(_, _, rgba, _, _) = self._p.getCameraImage(
			width=width,
			height=height,
			viewMatrix=view_matrix,
			projectionMatrix=proj_matrix,
			renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
			flags=pybullet.ER_NO_SEGMENTATION_MASK,
		)
		return rgba[..., :3]

	def close(self):
		self._p.disconnect()

	def is_robot_colliding(self):
		contact_points = self._p.getContactPoints(self.robot.robot_id)
		return len(contact_points) > 0

	def is_robot_falling(self):
		linear_velocity, _ = self._p.getBaseVelocity(self.robot.robot_id)
		return linear_velocity[2] < -5.

	def is_object_colliding(self):
		for body_id in range(self._p.getNumBodies()):
			# Skip robot id
			if body_id == self.robot.robot_id:
				continue
			# Check if object is exploding
			linear_velocity, _ = self._p.getBaseVelocity(body_id)
			if np.linalg.norm(linear_velocity) > 20.:
				return True
		return False

	def visualize(self):
		self.reset()
		render_config = self.get_render_config()
		renders = []
		renders3p = []
		renders.append(self.render(**render_config))
		renders3p.append(self.render3p())
		for _ in range(200):
			self.step(self.action_space.sample())
			renders.append(self.render(**render_config))
			renders3p.append(self.render3p())
		return renders, renders3p
