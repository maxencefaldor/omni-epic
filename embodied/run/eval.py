import time
import pickle
import re
from collections import defaultdict
from functools import partial as bind
import cloudpickle
from datetime import datetime

import embodied
import numpy as np
from .. import distr


class Driver:

  def __init__(self, make_env_fns, parallel=True, height=720, width=1280, **kwargs):
    assert len(make_env_fns) >= 1
    self.parallel = parallel
    self.height = height
    self.width = width
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    if parallel:
      import multiprocessing as mp
      context = mp.get_context()
      self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
      fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
      self.procs = [
          distr.StoppableProcess(self._env_server, i, pipe, fn, start=True)
          for i, (fn, pipe) in enumerate(zip(fns, pipes))]
      self.pipes[0].send(('act_space',))
      self.act_space = self._receive(self.pipes[0])
    else:
      self.envs = [fn() for fn in make_env_fns]
      self.act_space = self.envs[0].act_space
    self.callbacks = []
    self.done = np.full((self.length,), False)
    self.acts = None
    self.carry = None
    self.reset()

  def reset(self, init_policy=None):
    self.done = np.full((self.length,), False)
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)

  def close(self):
    if self.parallel:
      [proc.stop() for proc in self.procs]
    else:
      [env.close() for env in self.envs]

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    acts = self.acts
    assert all(len(x) == self.length for x in acts.values())
    assert all(isinstance(v, np.ndarray) for v in acts.values())
    acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
    if self.parallel:
      [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
      obs = [self._receive(pipe) for pipe in self.pipes]
      [pipe.send(('get_success',)) for pipe in self.pipes]
      success = [self._receive(pipe) for pipe in self.pipes]
      [pipe.send(('render', self.height, self.width,)) for pipe in self.pipes]
      render = [self._receive(pipe) for pipe in self.pipes]
      [pipe.send(('render3p', self.height, self.width,)) for pipe in self.pipes]
      render3p = [self._receive(pipe) for pipe in self.pipes]
    else:
      obs = [env.step(act) for env, act in zip(self.envs, acts)]
      success = [env.get_success() for env in self.envs]
      render = [env.render(height=self.height, width=self.width) for env in self.envs]
      render3p = [env.render3p(height=self.height, width=self.width) for env in self.envs]
    obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}
    assert all(len(x) == self.length for x in obs.values()), obs
    acts, outs, self.carry = policy(obs, self.carry, **self.kwargs)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self.acts = acts
    trans = {**obs, **acts, **outs, 'success': success, 'render': render, 'render3p': render3p,}
    for i in range(self.length):
      trn = {k: v[i] for k, v in trans.items()}
      if not self.done[i]:
        [fn(trn, i, **self.kwargs) for fn in self.callbacks]
    step += len(obs['is_first']) - self.done.sum()
    episode += (obs['is_last'] & ~self.done).sum()
    self.done |= obs['is_last']
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _receive(self, pipe):
    try:
      msg, arg = pipe.recv()
      if msg == 'error':
        raise RuntimeError(arg)
      assert msg == 'result'
      return arg
    except Exception:
      print('Terminating workers due to an exception.')
      [proc.kill() for proc in self.procs]
      raise

  @staticmethod
  def _env_server(context, envid, pipe, ctor):
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while context.running:
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          obs = env.step(act)
          pipe.send(('result', obs))
        elif msg == 'obs_space':
          assert len(args) == 0
          pipe.send(('result', env.obs_space))
        elif msg == 'act_space':
          assert len(args) == 0
          pipe.send(('result', env.act_space))
        elif msg == 'get_success':
          assert len(args) == 0
          pipe.send(('result', env.get_success()))
        elif msg == 'render':
          assert len(args) == 2
          height = args[0]
          width = args[1]
          pipe.send(('result', env.render(height=height, width=width)))
        elif msg == 'render3p':
          assert len(args) == 2
          height = args[0]
          width = args[1]
          pipe.send(('result', env.render3p(height=height, width=width)))
        else:
          raise ValueError(f'Invalid message {msg}')
    except Exception as e:
      distr.warn_remote_error(e, f'Env{envid}')
      pipe.send(('error', e))
    finally:
      print(f'Closing env {envid}')
      env.close()
      pipe.close()


def eval(make_agent, make_env, args, num_episodes, height=720, width=1280, eval_dir=None):
  assert args.from_checkpoint

  agent = make_agent()

  logdir = embodied.Path(args.logdir)
  if eval_dir is None:
    eval_dir = logdir / 'eval/'
  eval_dir.mkdir()
  step = embodied.Counter()
  episodes = defaultdict(embodied.Agg)
  policy_fps = embodied.FPS()

  @embodied.timer.section('log_step')
  def log_step(tran, worker):

    episode = episodes[worker]
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    episode.add('success', tran['success'], agg='stack')

    if tran['is_first']:
      episode.reset()

    if worker < num_episodes:
      for key in args.log_keys_video:
        if key in tran:
          episode.add(f'policy_{key}', tran[key], agg='stack')
      if 'render' in tran:
        episode.add(f'policy_render', tran['render'], agg='stack')
      if 'render3p' in tran:
        episode.add(f'policy_render3p', tran['render3p'], agg='stack')
    for key, value in tran.items():
      if re.match(args.log_keys_sum, key):
        episode.add(key, value, agg='sum')
      if re.match(args.log_keys_avg, key):
        episode.add(key, value, agg='avg')
      if re.match(args.log_keys_max, key):
        episode.add(key, value, agg='max')

    if tran['is_last']:
      result = episode.result()
      timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
      with open(eval_dir / f"episode_{timestamp}_{result['length']}.pickle", "wb") as file:
        pickle.dump(result, file)

  fns = [bind(make_env, i) for i in range(args.num_envs)]
  driver = Driver(fns, args.driver_parallel, height=height, width=width)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(log_step)

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation')
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  driver(policy, episodes=num_episodes)
  driver.close()
  print(f'Steps: {step.value}')
  print(f'Policy FPS: {policy_fps.result()}')
