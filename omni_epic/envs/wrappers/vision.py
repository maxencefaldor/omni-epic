import gym


class VisionWrapper(gym.Wrapper):

	def __init__(self, env, height=64, width=64, use_depth=True, fov=90.):
		super().__init__(env)
		self._height = height
		self._width = width
		self._use_depth = use_depth
		self._fov = fov

	def vision(self):
		return self.robot.vision(self._height, self._width, self._use_depth, self._fov)
