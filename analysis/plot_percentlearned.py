import os
import json
import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from analysis.visualize_taskgen import read_last_json_entry

def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, confidence_level=0.95):
    n = len(data)
    bootstrap_samples = np.random.choice(data, (num_bootstrap_samples, n), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    lower_bound = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)
    return lower_bound, upper_bound

@hydra.main(version_base=None, config_path="../configs/", config_name="plot_percentlearned")
def main(config: DictConfig):
    total_trainings = []
    total_learned = []

    # Go through each archive path
    for archive_path in config.paths:
        archive = read_last_json_entry(archive_path)
        codepaths = archive['codepaths']
        failedtrain = archive['failedtrain']

        all_paths = codepaths + failedtrain
        all_paths = sorted(all_paths, key=lambda x: os.path.basename(os.path.dirname(x)))

        xs = np.arange(1, len(all_paths) + 1)
        ys = np.array([1 if codepath in codepaths else 0 for codepath in all_paths])
        ys = np.cumsum(ys)

        total_trainings.append(xs)
        total_learned.append(ys)

    # Truncate xss and yss to the minimum length
    min_length = min(len(xs) for xs in total_trainings)
    total_trainings = [xs[:min_length] for xs in total_trainings]
    total_learned = [ys[:min_length] for ys in total_learned]

    # Convert to numpy arrays for easier manipulation
    total_trainings = np.array(total_trainings)
    total_learned = np.array(total_learned)

    # Calculate the mean and bootstrap confidence intervals
    median_learned = np.median(total_learned, axis=0)
    median_percentage_learned = median_learned / total_trainings[0] * 100
    lower_bounds = []
    upper_bounds = []
    for i in range(min_length):
        lower, upper = bootstrap_confidence_interval(total_learned[:, i] / total_trainings[:, i] * 100)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # Save the data
    with open('percent_learned.json', 'w') as f:
        json.dump({
            'median_percentage_learned': median_percentage_learned.tolist(),
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'total_trainings': total_trainings.tolist(),
            'num_learned': total_learned.tolist(),
        }, f)

    # Plot the data with bootstrap confidence intervals
    plt.figure(figsize=(10, 6))
    plt.plot(total_trainings[0], median_percentage_learned, label='Median Percentage Learned')
    plt.fill_between(total_trainings[0], lower_bounds, upper_bounds, color='b', alpha=0.2, label='95% Confidence Interval')
    plt.title('Percentage of Tasks Learned over Training Iterations')
    plt.xlabel('Total Number of Trainings (Successful and Failed)')
    plt.ylabel('Percentage of Tasks Learned')
    plt.legend()
    plt.grid(True)
    plot_path = 'plot_percent_learned.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
