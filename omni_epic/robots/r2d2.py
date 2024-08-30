import functools

import numpy as np
from scipy.spatial.transform import Rotation
import gym.spaces

from omni_epic.robots.base import URDFRobot, angle_between_vectors_3d


class R2D2Robot(URDFRobot):

	wheel_list = ["right_front_wheel", "left_front_wheel", "right_back_wheel", "left_back_wheel"]

	angular_velocity_gain = 200.

	linear_velocity_delta = 0.2
	linear_velocity_max = 10.0

	angular_velocity_delta = 1.
	angular_velocity_max = 2 * 2 * np.pi

	jump_velocity = 5.0

	def __init__(self, bullet_client):
		urdf = "/workspace/src/omni_epic/robots/assets/r2d2.urdf"
		super().__init__(bullet_client, urdf, base_position=[0., 0., 0.5], base_orientation=[0., 0., np.sqrt(2)/2, -np.sqrt(2)/2], self_collision=False)

	@functools.cached_property
	def action_space(self):
		# Four actions: do nothing, move forward, move backward, rotate clockwise, rotate counterclockwise, jump
		return gym.spaces.Discrete(6)

	@functools.cached_property
	def observation_space(self):
		high = np.inf * np.ones((5,), dtype=np.float32)
		return gym.spaces.Box(-high, high, dtype=np.float32)

	def reset(self, seed=None):
		super().reset(seed=seed)

		self.links["base"].set_position_and_orientation(
			self.links["base"].position_init,
			self.links["base"].orientation_init,
		)
		self.links["base"].set_linear_velocity_and_angular_velocity(
			self.links["base"].linear_velocity_init,
			self.links["base"].angular_velocity_init,
		)
		for joint in self.joints.values():
			joint.reset_position_and_velocity(self.np_random.uniform(low=-0.01, high=0.01), 0.)
		self.update()

	def apply_action(self, action):
		base_rotation_init = Rotation.from_quat(self.links["base"].orientation_init)
		base_rotation = Rotation.from_quat(self.links["base"].orientation)
		base_rotation_relative = base_rotation * base_rotation_init.inv()
		up_vector = base_rotation_relative.apply(np.array([0., 0., 1.]))
		cross_vector = np.cross(up_vector, np.array([0., 0., 1.]))
		angle = angle_between_vectors_3d(up_vector, np.array([0., 0., 1.]))

		if action == 0:
			# Do nothing
			new_angular_velocity = np.clip([0., 0., self.links["base"].angular_velocity[2]] + self.angular_velocity_gain * angle * cross_vector, -self.angular_velocity_max, self.angular_velocity_max)
			self._p.resetBaseVelocity(self.robot_id, angularVelocity=new_angular_velocity)
		if action == 1:
			# Go forward - up arrow
			forward_direction = base_rotation_relative.apply(np.array([1., 0., 0.]))
			new_linear_velocity = self.links["base"].linear_velocity + self.linear_velocity_delta * forward_direction
			normalize = min(self.linear_velocity_max / np.linalg.norm(new_linear_velocity), 1.)
			new_angular_velocity = np.clip([0., 0., self.links["base"].angular_velocity[2]] + self.angular_velocity_gain * angle * cross_vector, -self.angular_velocity_max, self.angular_velocity_max)
			self._p.resetBaseVelocity(self.robot_id, linearVelocity=normalize * new_linear_velocity, angularVelocity=new_angular_velocity)
		if action == 2:
			# Go backward - down arrow
			forward_direction = base_rotation_relative.apply(np.array([1., 0., 0.]))
			new_linear_velocity = self.links["base"].linear_velocity - self.linear_velocity_delta * forward_direction
			normalize = min(self.linear_velocity_max / np.linalg.norm(new_linear_velocity), 1.)
			new_angular_velocity = np.clip([0., 0., self.links["base"].angular_velocity[2]] + self.angular_velocity_gain * angle * cross_vector, -self.angular_velocity_max, self.angular_velocity_max)
			self._p.resetBaseVelocity(self.robot_id, linearVelocity=normalize * new_linear_velocity, angularVelocity=new_angular_velocity)
		if action == 3:
			# Rotate clockwise - right arrow
			new_angular_velocity = np.clip([0., 0., self.links["base"].angular_velocity[2]] - self.angular_velocity_delta * up_vector + self.angular_velocity_gain * angle * cross_vector, -self.angular_velocity_max, self.angular_velocity_max)
			self._p.resetBaseVelocity(self.robot_id, angularVelocity=new_angular_velocity)
		if action == 4:
			# Rotate counterclockwise - left arrow
			new_angular_velocity = np.clip([0., 0., self.links["base"].angular_velocity[2]] + self.angular_velocity_delta * up_vector + self.angular_velocity_gain * angle * cross_vector, -self.angular_velocity_max, self.angular_velocity_max)
			self._p.resetBaseVelocity(self.robot_id, angularVelocity=new_angular_velocity)
		if action == 5:
			# Jump - space
			wheel_contact = self._get_wheel_contact()
			is_standing = np.any(wheel_contact, keepdims=True).astype(np.float32)
			if not is_standing:
				new_angular_velocity = np.clip([0., 0., self.links["base"].angular_velocity[2]] + self.angular_velocity_gain * angle * cross_vector, -self.angular_velocity_max, self.angular_velocity_max)
				self._p.resetBaseVelocity(self.robot_id, angularVelocity=new_angular_velocity)
			else:
				new_linear_velocity = np.array([self.links["base"].linear_velocity[0], self.links["base"].linear_velocity[1], self.jump_velocity])  # no clipping
				new_angular_velocity = np.clip([0., 0., self.links["base"].angular_velocity[2]] + self.angular_velocity_gain * angle * cross_vector, -self.angular_velocity_max, self.angular_velocity_max)
				self._p.resetBaseVelocity(self.robot_id, linearVelocity=new_linear_velocity)

	def get_observation(self):
		qvel = np.concatenate([self.links["base"].linear_velocity, self.links["base"].angular_velocity[2:]])  # (4,)
		wheel_contact = self._get_wheel_contact()
		is_standing = np.any(wheel_contact, keepdims=True).astype(np.float32)  # (1,)
		return np.concatenate([qvel, is_standing])

	def get_rewards(self, action):
		if action == 0:
			return {"energy_penalty": 0.}
		elif action == 1 or action == 2:
			return {"energy_penalty": -0.2}
		elif action == 3 or action == 4:
			return {"energy_penalty": -0.1}
		elif action == 5:
			return {"energy_penalty": -0.5}

	def _get_wheel_contact(self):
		return np.asarray([len(self._p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.links[wheel].index)) > 0 for wheel in self.wheel_list], dtype=np.float32)

	def _get_eye_target_up(self):
		head_rotation = Rotation.from_quat(self.links["base"].orientation)

		# Eye position
		eye_position = self.links["base"].position + head_rotation.apply(np.array([0., 0., 0.3]) + np.array([0., 0.1214, 0.1214]))  # head is 0.25m in front of the torso

		# Target position
		target_position = eye_position + 10. * head_rotation.apply(np.array([0., 1., 0.]))

		# Up vector
		up_vector = head_rotation.apply(np.array([0., 0., 1.]))

		return eye_position, target_position, up_vector
