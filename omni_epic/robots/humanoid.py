import os
import functools

import numpy as np
from scipy.spatial.transform import Rotation
import gym.spaces
import pybullet
import pybullet_data

from omni_epic.robots.base import MJCFRobot, angle_between_vectors_3d


class HumanoidRobot(MJCFRobot):

	electricity_penalty_weight = 0.85
	stall_torque_penalty_weight = 0.425
	joints_at_limit_penalty_weight = 0.1

	foot_list = ["left_foot", "right_foot"]
	joints_torque_max = {
		"abdomen_z": 41., "abdomen_y": 41., "abdomen_x": 41.,
		"right_hip_x": 41., "right_hip_z": 41., "right_hip_y": 123., "right_knee": 82.,
		"left_hip_x":  41., "left_hip_z": 41., "left_hip_y": 123., "left_knee": 82.,
		"right_shoulder1": 30.75, "right_shoulder2": 30.75, "right_elbow": 30.75,
		"left_shoulder1": 30.75, "left_shoulder2": 30.75, "left_elbow": 30.75,
	}

	def __init__(self, bullet_client):
		mjcf = os.path.join(pybullet_data.getDataPath(), "mjcf", "humanoid_symmetric.xml")
		super().__init__(bullet_client, mjcf, self_collision=True, joints_torque_max=self.joints_torque_max)

	@functools.cached_property
	def action_space(self):
		high = np.ones((17,), dtype=np.float32)
		return gym.spaces.Box(-high, high, dtype=np.float32)

	@functools.cached_property
	def observation_space(self):
		high = np.inf * np.ones((46,), dtype=np.float32)
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
		torque = self._action_to_torque(action)
		self._p.setJointMotorControlArray(
			bodyIndex=self.robot_id,
			jointIndices=self._joints_index,
			controlMode=pybullet.TORQUE_CONTROL,
			forces=torque,
		)

	def _action_to_torque(self, action):
		return self._joints_torque_max * np.clip(action, -1., +1.)

	def get_observation(self):
		qpos = np.concatenate([self.links["base"].orientation] + [joint.position_norm for joint in self.joints.values()])  # (21,)
		qvel = np.concatenate([self.links["base"].linear_velocity, self.links["base"].angular_velocity] + [joint.velocity for joint in self.joints.values()])  # (23,)
		feet_contact = self._get_feet_contact()  # (2,)
		return np.concatenate([qpos, qvel, feet_contact])

	def get_rewards(self, action):
		# Energy penalty
		joints_velocity = np.array([joint.velocity[0] for joint in self.joints.values()])
		electricity_penalty = self.electricity_penalty_weight * float(np.abs(action * joints_velocity).mean())
		stall_torque_penalty = self.stall_torque_penalty_weight * float(np.square(action).mean())

		# Joints at limit penalty
		joints_position_norm = np.array([joint.position_norm[0] for joint in self.joints.values()])
		joints_at_limit = np.count_nonzero(np.abs(joints_position_norm) > 0.99)
		joints_at_limit_penalty = self.joints_at_limit_penalty_weight * float(joints_at_limit)

		return {"electricity_penalty": -electricity_penalty, "stall_torque_penalty": -stall_torque_penalty, "joints_at_limit_penalty": -joints_at_limit_penalty}

	def _get_feet_contact(self):
		return np.asarray([len(self._p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.links[foot].index)) > 0 for foot in self.foot_list], dtype=np.float32)

	def _get_contact_force(self):
		forces = []
		for foot in self.foot_list:
			contact_points = self._p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.links[foot].index)
			if contact_points:
				contact_normal = np.sum(np.asarray([contact_point[9] * np.asarray(contact_point[7]) for contact_point in contact_points], dtype=np.float32), axis=0)
				lateral_friction_1 = np.sum(np.asarray([contact_point[10] * np.asarray(contact_point[11]) for contact_point in contact_points], dtype=np.float32), axis=0)
				lateral_friction_2 = np.sum(np.asarray([contact_point[12] * np.asarray(contact_point[13]) for contact_point in contact_points], dtype=np.float32), axis=0)
				force = contact_normal + lateral_friction_1 + lateral_friction_2
			else:
				force = np.zeros(3, dtype=np.float32)
			forces.append(force)
		return np.concatenate(forces, axis=0)

	def _get_eye_target_up(self):
		head_rotation = Rotation.from_quat(self.links["base"].orientation)

		# Eye position
		eye_position = self.links["base"].position + head_rotation.apply(np.array([0.09, 0., 0.19]))  # head is 0.19m above the torso

		# Target position
		target_position = eye_position + 10. * head_rotation.apply(np.array([1., 0., 0.]))

		# Up vector
		up_vector = head_rotation.apply(np.array([0., 0., 1.]))

		return eye_position, target_position, up_vector
