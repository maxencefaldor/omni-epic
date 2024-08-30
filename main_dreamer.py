import warnings
from functools import partial as bind

import dreamerv3
import embodied

import hydra
from omegaconf import OmegaConf, DictConfig

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')


@hydra.main(version_base=None, config_path="configs/dreamer/", config_name="dreamer_xxs")
def main_dreamer(config: DictConfig) -> None:
	config = embodied.Config(OmegaConf.to_container(config))
	config, _ = embodied.Flags(config).parse_known()

	def make_logger(config):
		logdir = embodied.Path(config.logdir)
		logger_list = [
			embodied.logger.TerminalOutput(),
			embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
		]
		if config.wandb:
			logger_list.append(embodied.logger.WandBOutput(logdir, config=config))
		return embodied.Logger(embodied.Counter(), logger_list)

	def make_env(config, env_id=0):
		from embodied.envs.pybullet import PyBullet
		env = PyBullet(config.env.path, vision=config.env.vision, size=config.env.size, use_depth=config.env.use_depth, fov=config.env.fov)
		env = dreamerv3.wrap_env(env, config)
		return env

	def make_replay(config):
		return embodied.replay.Replay(
				length=config.batch_length,
				capacity=config.replay.size,
				directory=embodied.Path(config.logdir) / 'replay',
				online=config.replay.online,
		)

	def make_agent(config):
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

	embodied.run.train(
			bind(make_agent, config),
			bind(make_replay, config),
			bind(make_env, config),
			bind(make_logger, config),
			args,
	)


if __name__ == "__main__":
	main_dreamer()
