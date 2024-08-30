import functools
import importlib.util

import embodied
import numpy as np

from omni_epic.envs.wrappers.vision import VisionWrapper


class PyBullet(embodied.Env):

  def __init__(self, env_path, vision=True, size=(64, 64), use_depth=True, fov=90.):
    spec = importlib.util.spec_from_file_location('env', env_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Env = getattr(module, 'Env')
    self._env = Env()
    if vision:
      self._env = VisionWrapper(self._env, height=size[0], width=size[1], use_depth=use_depth, fov=fov)
    self._vision = vision
    self._size = size
    self._use_depth = use_depth
    self._terminated = True
    self._info = None

  @functools.cached_property
  def obs_space(self):
    obs_space = {
        'vector': self._convert(self._env.observation_space),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
      }
    if self._vision:
      obs_space['image'] = embodied.Space(np.float32, self._size + (4,), low=0., high=1.) if self._use_depth else embodied.Space(np.float32, self._size + (3,), low=0., high=1.)
    return obs_space

  @functools.cached_property
  def act_space(self):
    return {
        'action': self._convert(self._env.action_space),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._terminated:
      self._terminated = False
      obs = self._env.reset()
      obs = {
          'vector': obs,
          'reward': np.float32(0.0),
          'is_first': True,
          'is_last': False,
          'is_terminal': False,
      }
    else:
      obs, reward, self._terminated, truncated, self._info = self._env.step(action['action'])
      obs = {
          'vector': obs,
          'reward': np.float32(reward),
          'is_first': False,
          'is_last': self._terminated or truncated,
          'is_terminal': self._terminated,
      }
    if self._vision:
      obs['image'] = self._env.vision()
    return obs

  def get_success(self):
    return self._env.get_success()

  def render(self, *args, **kwargs):
      return self._env.render(*args, **kwargs)

  def render3p(self, *args, **kwargs):
      return self._env.render3p(*args, **kwargs)

  def close(self):
    self._env.close()

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)
