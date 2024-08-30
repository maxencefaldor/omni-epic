import logging

import numpy as np
from gym.utils import seeding
import pybullet
logger = logging.getLogger(__name__)


class Robot:
	"""
	Abstract class for robots.
	"""

	action_space: ...
	observation_space: ...

	_np_random = None

	@property
	def np_random(self):
		if self._np_random is None:
			self._np_random, seed = seeding.np_random()
		return self._np_random

	@np_random.setter
	def np_random(self, value: np.random.Generator):
		self._np_random = value

	def reset(self, seed=None):
		if seed is not None:
			self._np_random, seed = seeding.np_random(seed)

	def apply_action(self, action):
		raise NotImplementedError

	def get_observation(self):
		raise NotImplementedError

	def get_rewards(self, action):
		raise NotImplementedError

	def vision(self, height, width, use_depth, fov):
		near = 0.01
		far = 100.
		eye_position, target_position, up_vector = self._get_eye_target_up()
		view_matrix = self._p.computeViewMatrix(
			cameraEyePosition=eye_position,
			cameraTargetPosition=target_position,
			cameraUpVector=up_vector,
		)
		proj_matrix = self._p.computeProjectionMatrixFOV(
			fov=fov,
			aspect=width / height,
			nearVal=near,
			farVal=far,
		)
		(_, _, rgba, zbuffer, _) = self._p.getCameraImage(
			width=width,
			height=height,
			viewMatrix=view_matrix,
			projectionMatrix=proj_matrix,
			renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
			flags=pybullet.ER_NO_SEGMENTATION_MASK,
		)
		if use_depth:
			def zbuffer_to_depth(zbuffer):
				return far * near / (far - (far - near) * zbuffer)

			depth = (zbuffer_to_depth(zbuffer) - near) / far
			return np.concatenate([rgba[..., :3] / 255., depth[..., None]], axis=-1, dtype=np.float32)
		else:
			return (rgba[..., :3] / 255.).astype(np.float32)

class XMLRobot(Robot):
	"""
	Abstract class for XML based robots.
	"""

	def __init__(self, bullet_client):
		self._p = bullet_client

	def _init(self, robot_id):
		links, joints = {}, {}
		links["base"] = Base(self._p, robot_id)
		for joint_index in range(self._p.getNumJoints(bodyUniqueId=robot_id)):
			self._p.setJointMotorControl2(
				bodyUniqueId=robot_id,
				jointIndex=joint_index,
				controlMode=pybullet.POSITION_CONTROL,
				force=0,
				positionGain=0.1,
				velocityGain=0.1,
			)  # TODO: is it possible to use disable method of Joint class?
			joint_info = self._p.getJointInfo(bodyUniqueId=robot_id, jointIndex=joint_index)
			joint_name = joint_info[1].decode("utf8")
			link_name = joint_info[12].decode("utf8")

			assert link_name not in links, f"Link {link_name} already exists in links dictionary."
			links[link_name] = Link(self._p, robot_id, joint_index, link_name)

			if joint_name.startswith("ignore"):
				logger.info(f"Ignore joint {joint_name}.")
				Joint(self._p, robot_id, joint_index, joint_name).disable()
				continue
			elif joint_name.startswith("jointfix"):
				logger.info(f"Ignore joint {joint_name}.")
			elif joint_info[2] == pybullet.JOINT_FIXED:
				logger.info(f"Ignore joint {joint_name}.")
			else:
				assert joint_name not in joints, f"Joint {joint_name} already exists in joints dictionary."
				assert joint_info[2] == pybullet.JOINT_REVOLUTE, f"Joint {joint_name} is not supported."
				joints[joint_name] = Joint(self._p, robot_id, joint_index, joint_name)
		return links, joints


class URDFRobot(XMLRobot):
	"""
	Base class for URDF based robots.
	"""

	def __init__(self, bullet_client, urdf, base_position=[0., 0., 0.], base_orientation=[0., 0., 0., 1.], fixed_base=False, self_collision=True):
		super().__init__(bullet_client)
		self.base_position = base_position
		self.base_orientation = base_orientation
		self.fixed_base = fixed_base
		self.self_collision = self_collision

		# Load URDF
		if self_collision:
			flags = pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_GOOGLEY_UNDEFINED_COLORS
		else:
			flags = pybullet.URDF_GOOGLEY_UNDEFINED_COLORS
		self.robot_id = self._p.loadURDF(
			fileName=urdf,
			basePosition=self.base_position,
			baseOrientation=self.base_orientation,
			useFixedBase=self.fixed_base,
			flags=flags,
		)
		self.links, self.joints = self._init(self.robot_id)

		# Mass
		self.mass = np.asarray([link.mass for link in self.links.values()]).sum()

		# Links
		self._link_index = np.asarray([link.index for link in self.links.values()], dtype=np.int32)

		# Joints
		self._joints_index = np.asarray([joint.index for joint in self.joints.values()], dtype=np.int32)

	def _update_links(self):
		base, *links_list = self.links.values()

		# Update base
		base.position, base.orientation = base._get_position_and_orientation()
		base.linear_velocity, base.angular_velocity = base._get_linear_velocity_and_angular_velocity()

		# Update links
		links_state = self._p.getLinkStates(bodyUniqueId=self.robot_id, linkIndices=self._link_index[1:], computeLinkVelocity=1)
		for link, link_state in zip(links_list, links_state):
			link.position, link.orientation = np.asarray(link_state[0], dtype=np.float32), np.asarray(link_state[1], dtype=np.float32)
			link.linear_velocity, link.angular_velocity = np.asarray(link_state[6], dtype=np.float32), np.asarray(link_state[7], dtype=np.float32)

	def _update_joints(self):
		if len(self.joints) > 0:
			joints_state = self._p.getJointStates(bodyUniqueId=self.robot_id, jointIndex=self._joints_index)
			for joint, joint_state in zip(self.joints.values(), joints_state):
				joint.position, joint.velocity = np.asarray([joint_state[0]], dtype=np.float32), np.asarray([joint_state[1]], dtype=np.float32)

	def update(self):
		self._update_links()
		self._update_joints()


class MJCFRobot(XMLRobot):
	"""
	Base class for MJCF based robots.
	"""

	def __init__(self, bullet_client, mjcf, self_collision=True, joints_torque_max=None):
		super().__init__(bullet_client)
		self.self_collision = self_collision

		# Load MJCF
		if self_collision:
			self.flags = pybullet.URDF_USE_SELF_COLLISION | pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS | pybullet.URDF_GOOGLEY_UNDEFINED_COLORS
		else:
			self.flags = pybullet.URDF_GOOGLEY_UNDEFINED_COLORS
		(self.robot_id,) = self._p.loadMJCF(mjcfFileName=mjcf, flags=self.flags)
		self.links, self.joints = self._init(self.robot_id)

		# Mass
		self.mass = np.asarray([link.mass for link in self.links.values()]).sum()

		# Links
		self._link_index = np.asarray([link.index for link in self.links.values()], dtype=np.int32)

		# Joints
		self._joints_index = np.asarray([joint.index for joint in self.joints.values()], dtype=np.int32)
		if joints_torque_max is not None:
			assert joints_torque_max.keys() == self.joints.keys(), "joints_max_torque keys must match self.joints keys."
			for joint_name, joint in self.joints.items():
				joint.torque_max = joints_torque_max[joint_name]
			self._joints_torque_max = np.asarray([joint.torque_max for joint in self.joints.values()], dtype=np.float32)

	def _update_links(self):
		base, *links_list = self.links.values()

		# Update base
		base.position, base.orientation = base._get_position_and_orientation()
		base.linear_velocity, base.angular_velocity = base._get_linear_velocity_and_angular_velocity()

		# Update links
		links_state = self._p.getLinkStates(bodyUniqueId=self.robot_id, linkIndices=self._link_index[1:], computeLinkVelocity=1)
		for link, link_state in zip(links_list, links_state):
			link.position, link.orientation = np.asarray(link_state[0], dtype=np.float32), np.asarray(link_state[1], dtype=np.float32)
			link.linear_velocity, link.angular_velocity = np.asarray(link_state[6], dtype=np.float32), np.asarray(link_state[7], dtype=np.float32)

	def _update_joints(self):
		joints_state = self._p.getJointStates(bodyUniqueId=self.robot_id, jointIndices=self._joints_index)
		if joints_state is None:
			return
		for joint, joint_state in zip(self.joints.values(), joints_state):
			joint.position, joint.velocity = np.asarray([joint_state[0]], dtype=np.float32), np.asarray([joint_state[1]], dtype=np.float32)

	def update(self):
		self._update_links()
		self._update_joints()


class Link:

	def __init__(self, bullet_client, robot_id, link_index, link_name):
		self._p = bullet_client
		self.robot_id = robot_id
		self.index = link_index
		self.name = link_name
		self.mass = self._p.getDynamicsInfo(bodyUniqueId=self.robot_id, linkIndex=self.index)[0]

		self.position_init, self.orientation_init = self._get_position_and_orientation()
		self.linear_velocity_init, self.angular_velocity_init = self._get_linear_velocity_and_angular_velocity()

		self.position, self.orientation = self.position_init, self.orientation_init
		self.linear_velocity, self.angular_velocity = self.linear_velocity_init, self.angular_velocity_init

	def _get_position_and_orientation(self):
		position, orientation, _, _, _, _ = self._p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.index)
		return np.asarray(position, dtype=np.float32), np.asarray(orientation, dtype=np.float32)  # return position and orientation in world frame

	def _get_linear_velocity_and_angular_velocity(self):
		_, _, _, _, _, _, linear_velocity, angular_velocity = self._p.getLinkState(bodyUniqueId=self.robot_id, linkIndex=self.index, computeLinkVelocity=1)
		return np.asarray(linear_velocity, dtype=np.float32), np.asarray(angular_velocity, dtype=np.float32)  # return velocity in world frame


class Base(Link):

	def __init__(self, bullet_client, robot_id):
		super().__init__(bullet_client, robot_id, -1, "base")

		self.position_init, self.orientation_init = self._get_position_and_orientation()
		self.linear_velocity_init, self.angular_velocity_init = self._get_linear_velocity_and_angular_velocity()

		self.position, self.orientation = self.position_init, self.orientation_init
		self.linear_velocity, self.angular_velocity = self.linear_velocity_init, self.angular_velocity_init

	def _get_position_and_orientation(self):
		position, orientation = self._p.getBasePositionAndOrientation(bodyUniqueId=self.robot_id)
		return np.asarray(position, dtype=np.float32), np.asarray(orientation, dtype=np.float32)  # return position and orientation in world frame

	def set_position_and_orientation(self, position, orientation):
		self._p.resetBasePositionAndOrientation(bodyUniqueId=self.robot_id, posObj=position, ornObj=orientation)
		self.position, self.orientation = position, orientation

	def _get_linear_velocity_and_angular_velocity(self):
		linear_velocity, angular_velocity = self._p.getBaseVelocity(bodyUniqueId=self.robot_id)
		return np.asarray(linear_velocity, dtype=np.float32), np.asarray(angular_velocity, dtype=np.float32)  # return velocity in world frame

	def set_linear_velocity_and_angular_velocity(self, linear_velocity, angular_velocity):
		self._p.resetBaseVelocity(objectUniqueId=self.robot_id, linearVelocity=linear_velocity, angularVelocity=angular_velocity)
		self.linear_velocity, self.angular_velocity = linear_velocity, angular_velocity


class Joint:

	def __init__(self, bullet_client, robot_id, joint_index, joint_name):
		self._p = bullet_client
		self.robot_id = robot_id
		self.index = joint_index
		self.name = joint_name

		joint_info = self._p.getJointInfo(bodyUniqueId=self.robot_id, jointIndex=self.index)
		self.lower_limit = joint_info[8]
		self.upper_limit = joint_info[9]
		self.torque_max = None

		self.position, self.velocity = self._get_position_and_velocity()

	def _get_position_and_velocity(self):
		position, velocity, _, _ = self._p.getJointState(bodyUniqueId=self.robot_id, jointIndex=self.index)
		return np.asarray([position], dtype=np.float32), np.asarray([velocity], dtype=np.float32)

	def set_position_and_velocity(self, position, velocity):
		self._p.resetJointState(bodyUniqueId=self.robot_id, jointIndex=self.index, targetValue=position, targetVelocity=velocity)
		self.position, self.velocity = position, velocity

	def reset_position_and_velocity(self, position, velocity):
		self.set_position_and_velocity(position, velocity)
		self.disable()

	@property
	def position_norm(self):
		# Normalize joint position to [-1., 1.]
		pos_mid = 0.5 * (self.lower_limit + self.upper_limit)
		return  2 * (self.position - pos_mid) / (self.upper_limit - self.lower_limit)

	def set_position(self, position):
		self._p.setJointMotorControl2(
			bodyUniqueId=self.robot_id,
			jointIndex=self.index,
			controlMode=pybullet.POSITION_CONTROL,
			targetPosition=position,
		)

	def set_velocity(self, velocity):
		self._p.setJointMotorControl2(
			bodyUniqueId=self.robot_id,
			jointIndex=self.index,
			controlMode=pybullet.VELOCITY_CONTROL,
			targetVelocity=velocity,
		)

	def set_torque(self, torque):
		self._p.setJointMotorControl2(
			bodyUniqueId=self.robot_id,
			jointIndex=self.index,
			controlMode=pybullet.TORQUE_CONTROL,
			force=torque,
		)

	def disable(self):
		self._p.setJointMotorControl2(
			bodyUniqueId=self.robot_id,
			jointIndex=self.index,
			controlMode=pybullet.POSITION_CONTROL,
			targetPosition=0.,
			targetVelocity=0.,
			force=0.,
			positionGain=0.1,  # TODO: is it needed?
			velocityGain=0.1,  # TODO: is it needed?
		)


def angle_between_vectors_2d(v_1, v_2):
	return np.arctan2(np.cross(v_1, v_2), np.dot(v_1, v_2))

def angle_between_vectors_3d(v_1, v_2):
	return np.arctan2(np.linalg.norm(np.cross(v_1, v_2)), np.dot(v_1, v_2))
