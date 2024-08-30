from omni_epic.envs.base import Env
from omni_epic.robots.r2d2 import R2D2Robot


class R2D2Env(Env):
	dt = 0.0165

	def __init__(self):
		# Init world
		super().__init__()

		# Init robot
		self.robot = R2D2Robot(self._p)

		self.action_space = self.robot.action_space
		self.observation_space = self.robot.observation_space

	def get_truncated(self, action):
		return False
