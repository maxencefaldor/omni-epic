import os
import hydra
from omegaconf import DictConfig
import json
import numpy as np
from scipy.stats import bootstrap

from analysis.visualize_taskgen import read_last_json_entry


@hydra.main(version_base=None, config_path="../configs/", config_name="plot_envgen_success")
def main(config: DictConfig):
    # Trackers
    num_gens_totals = []
    num_gens_eventualsucc_totals = []
    num_gens_successfuls = []

    # Go through each run folder
    for run_folder in config.paths:
        archive = read_last_json_entry(os.path.join(run_folder, 'archive.jsonl'))
        eventualsucc_envpaths = archive['codepaths'] + archive['failedint'] + archive['failedtrain']
        num_gens_total = 0
        num_gens_eventualsucc_total = 0
        num_gens_successful = len(eventualsucc_envpaths)

        # Go through all folders in path with the name "task_*"
        for folder in os.listdir(run_folder):
            if not folder.startswith("task_"):
                continue

            # Get the number of env_*.py files in the folder
            env_files = [f for f in os.listdir(os.path.join(run_folder, folder)) if f.startswith("env_") and f.endswith(".py")]
            num_gens_total += len(env_files)

            # if any of the env_files is in eventualsucc_envpaths
            for env_file in env_files:
                if os.path.join(run_folder, folder, env_file) in eventualsucc_envpaths:
                    num_gens_eventualsucc_total += len(env_files)

        # Append trackers
        num_gens_totals.append(num_gens_total)
        num_gens_eventualsucc_totals.append(num_gens_eventualsucc_total)
        num_gens_successfuls.append(num_gens_successful)

    # Calculate metrics
    success_rates = [s / t for s, t in zip(num_gens_successfuls, num_gens_totals)]
    median_success_rate = np.median(success_rates)
    res = bootstrap((np.array(success_rates),), np.median, confidence_level=0.95, n_resamples=10000)
    confidence_interval = res.confidence_interval

    success_rates_eventualsucc = [s / t for s, t in zip(num_gens_successfuls, num_gens_eventualsucc_totals)]
    median_success_rate_eventualsucc = np.median(success_rates_eventualsucc)
    res_eventualsucc = bootstrap((np.array(success_rates_eventualsucc),), np.median, confidence_level=0.95, n_resamples=10000)
    confidence_interval_eventualsucc = res_eventualsucc.confidence_interval

    # Save results to json file
    results = {
        "num_gens_totals": num_gens_totals,
        "num_gens_eventualsucc_totals": num_gens_eventualsucc_totals,
        "num_gens_successfuls": num_gens_successfuls,

        "success_rates": success_rates,
        "median_success_rate": median_success_rate,
        "confidence_interval": [confidence_interval.low, confidence_interval.high],

        "success_rates_eventualsucc": success_rates_eventualsucc,
        "median_success_rate_eventualsucc": median_success_rate_eventualsucc,
        "confidence_interval_eventualsucc": [confidence_interval_eventualsucc.low, confidence_interval_eventualsucc.high],
    }
    with open('output.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to output.json")

if __name__ == "__main__":
    main()
