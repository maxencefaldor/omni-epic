import os
import hydra
from omegaconf import DictConfig

from analysis.visualize_taskgen import read_last_json_entry
from main_dreamer import main_dreamer


@hydra.main(version_base=None, config_path="../configs/", config_name="run_scratch")
def main(config: DictConfig):
    archive = read_last_json_entry(config.path)
    codepaths = archive['codepaths']
    codepaths = codepaths[:config.num_tasks] if config.num_tasks > 0 else codepaths
    config_dreamer = config.dreamer

    for task_envpath in codepaths:
        task_key = os.path.basename(os.path.dirname(task_envpath))
        task_dir = os.path.join(config.logdir, task_key)

        # Dreamer config
        dreamer_dir = os.path.join(task_dir, 'dreamer/')
        if os.path.exists(dreamer_dir):
            print(f"Skipping {task_key} as it already exists.")
            continue
        config_dreamer.logdir = dreamer_dir
        config_dreamer.env.path = task_envpath

        # Run Dreamer
        main_dreamer(config_dreamer)


if __name__ == "__main__":
    main()
