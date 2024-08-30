from pathlib import Path
from omegaconf import OmegaConf
import pickle

import dreamerv3
import embodied
from embodied.run.eval import eval

import mediapy


def visualize_episodes(eval_dir):
	for pickle_path in eval_dir.glob("*.pickle"):
		with open(str(pickle_path), "rb") as f:
			episode = pickle.load(f)
		
		print(pickle_path.stem)
		print(f"\tScore: {episode['score']}")
		print(f"\tLength: {episode['length']}")

		video_path = eval_dir / f"render_{pickle_path.stem}.mp4"
		if not video_path.is_file():
			mediapy.write_video(str(video_path), episode["policy_render"])

		video_path = eval_dir / f"render3p_{pickle_path.stem}.mp4"
		if not video_path.is_file():
			mediapy.write_video(str(video_path), episode["policy_render3p"])

		video_path = eval_dir / f"vision_{pickle_path.stem}.mp4"
		if not video_path.is_file():
			mediapy.write_video(str(video_path), [policy_image[..., :3] for policy_image in episode["policy_image"]])

		success_path = eval_dir / f"success_{pickle_path.stem}.txt"
		if not success_path.is_file():
			with open(str(success_path), "w") as f:
				f.write("\n".join([str(x) for x in episode["success"]]))


def visualize_dreamer(dreamer_dir, num_episodes=5) -> None:
	dreamer_dir = Path(dreamer_dir)
	try:
		config_dreamer = OmegaConf.load(dreamer_dir / ".hydra" / "config.yaml")
	except FileNotFoundError:
		config_dreamer = OmegaConf.load(dreamer_dir / ".." / ".." / ".hydra" / "config.yaml")
		config_dreamer = config_dreamer.dreamer
		if (dreamer_dir / ".." / "env_5.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_5.py")
		elif (dreamer_dir / ".." / "env_4.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_4.py")
		elif (dreamer_dir / ".." / "env_3.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_3.py")
		elif (dreamer_dir / ".." / "env_2.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_2.py")
		elif (dreamer_dir / ".." / "env_1.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_1.py")
		elif (dreamer_dir / ".." / "env_0.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_0.py")
		else:
			raise FileNotFoundError
	config = embodied.Config(OmegaConf.to_container(config_dreamer))
	config = config.update({
		"logdir": str(dreamer_dir),
		"jax.policy_devices": [0],
		"jax.train_devices": [0],
		"run.from_checkpoint": str(dreamer_dir / "checkpoint.ckpt"),
		"run.num_envs": num_episodes,
	})
	config, _ = embodied.Flags(config).parse_known()

	def make_env(env_id=0):
		from embodied.envs.pybullet import PyBullet
		env = PyBullet(config.env.path, vision=config.env.vision, size=config.env.size, fov=config.env.fov)
		env = dreamerv3.wrap_env(env, config)
		return env

	def make_agent():
		env = make_env(config)
		agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
		env.close()
		return agent

	args = embodied.Config(
			**config.run,
			logdir=config.logdir,
			batch_size=config.batch_size,
			batch_length=config.batch_length,
			batch_length_eval=config.batch_length_eval,
			replay_context=config.replay_context,
	)

	# Eval agent
	eval(make_agent, make_env, args, num_episodes=num_episodes)

	# Visualize episodes
	eval_dir = dreamer_dir / 'eval'
	visualize_episodes(eval_dir)


def visualize_ckpt_env(ckpt_dir, env_path, eval_dir, num_episodes=5) -> None:
	dreamer_dir = Path(ckpt_dir) / 'dreamer'
	try:
		config_dreamer = OmegaConf.load(dreamer_dir / ".hydra" / "config.yaml")
	except FileNotFoundError:
		config_dreamer = OmegaConf.load(dreamer_dir / ".." / ".." / ".hydra" / "config.yaml")
		config_dreamer = config_dreamer.dreamer
		if (dreamer_dir / ".." / "env_5.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_5.py")
		elif (dreamer_dir / ".." / "env_4.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_4.py")
		elif (dreamer_dir / ".." / "env_3.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_3.py")
		elif (dreamer_dir / ".." / "env_2.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_2.py")
		elif (dreamer_dir / ".." / "env_1.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_1.py")
		elif (dreamer_dir / ".." / "env_0.py"). is_file():
			config_dreamer.env.path = str(dreamer_dir / ".." / "env_0.py")
		else:
			raise FileNotFoundError
	config = embodied.Config(OmegaConf.to_container(config_dreamer))
	config = config.update({
		"logdir": str(dreamer_dir),
		"jax.policy_devices": [0],
		"jax.train_devices": [0],
		"run.from_checkpoint": str(dreamer_dir / "checkpoint.ckpt"),
		"run.num_envs": num_episodes,
	})
	config, _ = embodied.Flags(config).parse_known()

	def make_env(env_id=0):
		from embodied.envs.pybullet import PyBullet
		env = PyBullet(env_path, vision=config.env.vision, size=config.env.size, fov=config.env.fov)
		env = dreamerv3.wrap_env(env, config)
		return env

	def make_agent():
		env = make_env(config)
		agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
		env.close()
		return agent

	args = embodied.Config(
			**config.run,
			logdir=config.logdir,
			batch_size=config.batch_size,
			batch_length=config.batch_length,
			batch_length_eval=config.batch_length_eval,
			replay_context=config.replay_context,
	)

	# Eval agent
	eval_dir = Path(eval_dir)
	eval(make_agent, make_env, args, num_episodes=num_episodes, eval_dir=eval_dir)

	# Visualize episodes
	visualize_episodes(eval_dir)


if __name__ == "__main__":
	visualize_dreamer("/workspace/src/output/pipeline/2024-06-18_153134_574615/task_19/dreamer", num_episodes=4)
