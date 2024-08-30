import copy
import gc
import json
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

from omni_epic.robots import robot_dict
from analysis.visualize_taskgen import read_last_json_entry
from analysis.visualize_dreamer import visualize_ckpt_env
from run_utils import get_task_success_from_folder
from omni_epic.core.fm import FM
from rag_utils import get_similar_codepaths


def check_min_criterion(path, path_index, prev_paths, method_folder, n_prev_paths=-1, embedding_method='openai'):
    # Can the previous ckpts do the new task
    ckpt_successes = []

    # Get the previous paths
    if n_prev_paths > 0:
        prev_paths, prev_paths_indices = get_similar_codepaths(
            path,
            prev_paths,
            num_returns=n_prev_paths,
            embedding_method=embedding_method
        )

    # Evaluate previous ckpts on the new env task
    for j, env_path in zip(prev_paths_indices, prev_paths):
        eval_dir = f'./{method_folder}/eval_env{path_index}_ckpt{j}'
        ckpt_dir = os.path.dirname(env_path)
        if not os.path.exists(eval_dir):
            visualize_ckpt_env(ckpt_dir, path, eval_dir, num_episodes=5)
        task_success = get_task_success_from_folder(eval_dir, voting='any')
        ckpt_successes.append(task_success)
        # Clean up after each evaluation
        gc.collect()

    # Check if ckpt passes the min criterion
    return sum(ckpt_successes) <= len(ckpt_successes) / 2

def check_interesting(fm_moi, config, robot_desc, path, path_index, prev_paths, method_folder):
    json_file = f'./{method_folder}/eval_ckpt{path_index}.json'
    num_examples = config.model_of_interestingness.num_examples

    if not os.path.exists(json_file):
        # Query interestingness
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        moi_example_paths = copy.copy(prev_paths)
        if num_examples > 0 and len(moi_example_paths) > num_examples:
            moi_example_paths, _ = get_similar_codepaths(
                path,
                moi_example_paths,
                num_returns=num_examples,
                embedding_method=config.embedding_method,
            )
        _, is_interesting = fm_moi.query_interestingness(robot_desc, path, moi_example_paths)

        # Save the interestingness value as json
        json_data = {'is_interesting': is_interesting}
        with open(json_file, 'w') as f:
            json.dump(json_data, f)
    else:
        # Load the interestingness value from json
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            is_interesting = json_data['is_interesting']

    return is_interesting

def change_color(color, factor=0.5):
    """Darken the given color by the given factor."""
    # return tuple([max(0, min(c * factor, 1)) for c in color])
    return tuple([color[2], color[0], color[1]])

def plot_annecs_metrics(metrics_dict, config):
    colors = plt.cm.tab10.colors  # Get a set of colors to use for different methods
    method_color_map = {}  # Dictionary to store the color for each method

    # Find the minimum number of iterations across all methods
    min_iterations = min(len(metrics['median_annecs']) for metrics in metrics_dict.values())
    print(f"Minimum number of iterations: {min_iterations}")

    def apply_plot_customizations(ax, config, title=None):
        if config.remove_titles and title:
            ax.set_title("")
        if config.remove_axes_labels:
            ax.set_xlabel("")
            ax.set_ylabel("")
        if config.remove_legend:
            ax.legend().set_visible(False)
        else:
            ax.legend()

    # Combined plot
    plt.figure(figsize=(10, 6))

    for idx, (method, metrics) in enumerate(metrics_dict.items()):
        iterations = range(1, min_iterations + 1)
        median_annecs = metrics['median_annecs'][:min_iterations]
        lower_annecs = metrics['lower_annecs'][:min_iterations]
        upper_annecs = metrics['upper_annecs'][:min_iterations]
        median_annecs_omni = metrics['median_annecs_omni'][:min_iterations]
        lower_annecs_omni = metrics['lower_annecs_omni'][:min_iterations]
        upper_annecs_omni = metrics['upper_annecs_omni'][:min_iterations]

        if method not in method_color_map:
            method_color_map[method] = colors[idx % len(colors)]  # Assign a color to the method

        color = method_color_map[method]
        dark_color = change_color(color)

        # Plot ANNECS with confidence interval
        plt.plot(iterations, median_annecs, label=f'(ANNECS) {method}', linewidth=2, color=color)
        plt.fill_between(iterations, lower_annecs, upper_annecs, color=color, alpha=0.2)

        # Plot ANNECS-OMNI with dotted line and confidence interval
        plt.plot(iterations, median_annecs_omni, label=f'(ANNECS-OMNI) {method}', linestyle=':', linewidth=2, color=dark_color)
        plt.fill_between(iterations, lower_annecs_omni, upper_annecs_omni, color=dark_color, alpha=0.2)

    plt.xlabel('Completed Archive Size')
    plt.ylabel('Metric Value')
    plt.title('ANNECS and ANNECS-OMNI over Iterations for Different Methods')
    plt.grid(True)

    plt.xticks(range(1, min_iterations + 1, max(1, min_iterations // 10)))  # Set integer tick labels for x-axis
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Set integer tick labels for y-axis

    apply_plot_customizations(ax, config, 'ANNECS and ANNECS-OMNI over Iterations for Different Methods')

    plt.savefig(f'./plot_annecs_combined.{config.file_format}', bbox_inches='tight', transparent=config.remove_background)
    plt.close()

    # New combined plot for ANNECS only
    plt.figure(figsize=(10, 6))

    for idx, (method, metrics) in enumerate(metrics_dict.items()):
        iterations = range(1, min_iterations + 1)
        median_annecs = metrics['median_annecs'][:min_iterations]
        lower_annecs = metrics['lower_annecs'][:min_iterations]
        upper_annecs = metrics['upper_annecs'][:min_iterations]

        color = method_color_map[method]

        # Plot ANNECS with confidence interval
        plt.plot(iterations, median_annecs, label=f'{method}', linewidth=2, color=color)
        plt.fill_between(iterations, lower_annecs, upper_annecs, color=color, alpha=0.2)

    plt.xlabel('Completed Archive Size')
    plt.ylabel('ANNECS Value')
    plt.title('ANNECS over Iterations for Different Methods')
    plt.grid(True)

    plt.xticks(range(1, min_iterations + 1, max(1, min_iterations // 10)))
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    apply_plot_customizations(ax, config, 'ANNECS over Iterations for Different Methods')

    plt.savefig(f'./plot_annecs_combined1.{config.file_format}', bbox_inches='tight', transparent=config.remove_background)
    plt.close()

    # New combined plot for ANNECS-OMNI only
    plt.figure(figsize=(10, 6))

    for idx, (method, metrics) in enumerate(metrics_dict.items()):
        iterations = range(1, min_iterations + 1)
        median_annecs_omni = metrics['median_annecs_omni'][:min_iterations]
        lower_annecs_omni = metrics['lower_annecs_omni'][:min_iterations]
        upper_annecs_omni = metrics['upper_annecs_omni'][:min_iterations]

        color = method_color_map[method]

        # Plot ANNECS-OMNI with confidence interval
        plt.plot(iterations, median_annecs_omni, label=f'{method}', linestyle=':', linewidth=2, color=color)
        plt.fill_between(iterations, lower_annecs_omni, upper_annecs_omni, color=color, alpha=0.2)

    plt.xlabel('Completed Archive Size')
    plt.ylabel('ANNECS-OMNI Value')
    plt.title('ANNECS-OMNI over Iterations for Different Methods')
    plt.grid(True)

    plt.xticks(range(1, min_iterations + 1, max(1, min_iterations // 10)))
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    apply_plot_customizations(ax, config, 'ANNECS-OMNI over Iterations for Different Methods')

    plt.savefig(f'./plot_annecs_combined2.{config.file_format}', bbox_inches='tight', transparent=config.remove_background)
    plt.close()

    # Individual plots for each method
    for method, metrics in metrics_dict.items():
        plt.figure(figsize=(10, 6))
        iterations = range(1, min_iterations + 1)

        color = method_color_map[method]
        dark_color = change_color(color)

        # Plot ANNECS
        plt.plot(iterations, metrics['median_annecs'][:min_iterations], label='ANNECS', linewidth=2, color=color)
        plt.fill_between(iterations,
                         metrics['lower_annecs'][:min_iterations],
                         metrics['upper_annecs'][:min_iterations],
                         color=color, alpha=0.2)

        # Plot ANNECS-OMNI
        plt.plot(iterations, metrics['median_annecs_omni'][:min_iterations], label='ANNECS-OMNI', linestyle=':', linewidth=2, color=dark_color)
        plt.fill_between(iterations,
                         metrics['lower_annecs_omni'][:min_iterations],
                         metrics['upper_annecs_omni'][:min_iterations],
                         color=dark_color, alpha=0.2)

        plt.xlabel('Completed Archive Size')
        plt.ylabel('Metric Value')
        plt.title(f'ANNECS and ANNECS-OMNI over Iterations for {method}')
        plt.grid(True)

        plt.xticks(range(1, min_iterations + 1, max(1, min_iterations // 10)))  # Set integer tick labels for x-axis
        ax = plt.gca()
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Set integer tick labels for y-axis

        apply_plot_customizations(ax, config, f'ANNECS and ANNECS-OMNI over Iterations for {method}')

        plt.savefig(f'./plot_annecs_{method.replace("/", "")}.{config.file_format}', bbox_inches='tight', transparent=config.remove_background)
        plt.close()

    # Save numerical data to JSON file
    def serialize_data(data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            return data
        else:
            return float(data)  # Handle scalar values

    json_data = {
        method: {
            'iterations': list(range(1, min_iterations + 1)),
            **{key: serialize_data(metrics[key][:min_iterations])
            for key in metrics if key != 'iterations'}
        }
        for method, metrics in metrics_dict.items()
    }

    with open('annecs_data.json', 'w') as f:
        json.dump(json_data, f)

def bootstrap_ci(data, num_bootstrap_samples=10000, ci=95):
    bootstrapped_medians = []
    for _ in range(num_bootstrap_samples):
        resampled_data = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_medians.append(np.median(resampled_data))

    lower_percentile = (100 - ci) / 2
    upper_percentile = 100 - lower_percentile

    return (np.percentile(bootstrapped_medians, lower_percentile),
            np.percentile(bootstrapped_medians, upper_percentile))

def calculate_metrics(all_annecs, all_annecs_omni):
    num_iterations = min(len(run) for run in all_annecs)
    all_annecs = [run[:num_iterations] for run in all_annecs]
    all_annecs_omni = [run[:num_iterations] for run in all_annecs_omni]
    median_annecs = np.median(all_annecs, axis=0)
    median_annecs_omni = np.median(all_annecs_omni, axis=0)

    lower_annecs = []
    upper_annecs = []
    lower_annecs_omni = []
    upper_annecs_omni = []

    for i in range(num_iterations):
        annecs_at_i = [run[i] for run in all_annecs]
        annecs_omni_at_i = [run[i] for run in all_annecs_omni]

        lower, upper = bootstrap_ci(annecs_at_i)
        lower_annecs.append(lower)
        upper_annecs.append(upper)

        lower_omni, upper_omni = bootstrap_ci(annecs_omni_at_i)
        lower_annecs_omni.append(lower_omni)
        upper_annecs_omni.append(upper_omni)

    return {
        'median_annecs': median_annecs,
        'lower_annecs': lower_annecs,
        'upper_annecs': upper_annecs,
        'median_annecs_omni': median_annecs_omni,
        'lower_annecs_omni': lower_annecs_omni,
        'upper_annecs_omni': upper_annecs_omni,
    }

def process_method(config, robot_desc, method, method_config):
    all_annecs = []
    all_annecs_omni = []

    config_moi = config.model_of_interestingness
    fm_moi = FM(config_moi)

    codepaths_list = []
    for method_repeat, archive_path in enumerate(method_config.paths):
        data = read_last_json_entry(archive_path)
        codepaths = data['codepaths']
        codepaths_list.append(codepaths)

    # Find the minimum length of codepaths
    min_codepaths_length = min(len(codepaths) for codepaths in codepaths_list)

    for method_repeat, codepaths in enumerate(codepaths_list):
        # Values to plotted
        iterations = []
        annecs = []
        annecs_omni = []

        for i in range(min_codepaths_length):
            env_path = codepaths[i]
            prev_paths = codepaths[:i]

            # Check if the ckpt passes the min criterion and is interesting
            method_folder = f"{method.replace('/', '')}_{method_repeat}"
            if config.train_agent:
                passed_criterion = check_min_criterion(
                    env_path, i, prev_paths, method_folder,
                    n_prev_paths=config.num_prev_eval_envs,
                    embedding_method=config.embedding_method,
                )
            else:
                passed_criterion = True
            is_interesting = check_interesting(fm_moi, config, robot_desc, env_path, i, prev_paths, method_folder)

            # Update ANNECS values
            add_annecs = 1 if passed_criterion else 0
            curr_annecs = annecs[-1] + add_annecs if i > 0 else add_annecs

            # Update ANNECS-OMNI values
            add_annecs_omni = 1 if (passed_criterion and is_interesting) else 0
            curr_annecs_omni = annecs_omni[-1] + add_annecs_omni if i > 0 else add_annecs_omni

            # Append values
            iterations.append(i + 1)
            annecs.append(curr_annecs)
            annecs_omni.append(curr_annecs_omni)

        all_annecs.append(annecs)
        all_annecs_omni.append(annecs_omni)

    # Calculate metrics
    metrics = calculate_metrics(all_annecs, all_annecs_omni)

    return {
        'iterations': iterations,
        **metrics
    }

def significance_testing():
    # Load the annecs_data.json file
    with open('annecs_data.json', 'r') as f:
        data = json.load(f)
    
    # Extract data for comparisons
    methods = list(data.keys())

    # Prepare to store results
    comparison_results = {}

    # Compare each pair of methods
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]

            # Extract ANNECS and ANNECS-OMNI data for both methods
            median_annecs1 = data[method1]['median_annecs']
            median_annecs2 = data[method2]['median_annecs']
            median_annecs_omni1 = data[method1]['median_annecs_omni']
            median_annecs_omni2 = data[method2]['median_annecs_omni']

            # Perform t-test
            t_stat, p_val_annecs = stats.ttest_ind(median_annecs1, median_annecs2, equal_var=False)
            t_stat, p_val_annecs_omni = stats.ttest_ind(median_annecs_omni1, median_annecs_omni2, equal_var=False)

            # Perform Mann-Whitney U test
            u_stat, p_val_mannwhitney_annecs = stats.mannwhitneyu(median_annecs1, median_annecs2)
            u_stat, p_val_mannwhitney_annecs_omni = stats.mannwhitneyu(median_annecs_omni1, median_annecs_omni2)

            # Store the results
            comparison_results[f"{method1} vs {method2}"] = {
                't-test_annecs': p_val_annecs,
                't-test_annecs_omni': p_val_annecs_omni,
                'mannwhitney_annecs': p_val_mannwhitney_annecs,
                'mannwhitney_annecs_omni': p_val_mannwhitney_annecs_omni,
            }

    # Save results to a JSON file
    with open('significance_testing_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=4)

@hydra.main(version_base=None, config_path="../configs/", config_name="plot_annecs")
def main(config: DictConfig):
    robot_desc = robot_dict[config.robot]["robot_desc"]

    if not config.metrics_dict_path:
        # Process each method
        metrics_dict = {}
        for method, method_config in config.methods.items():
            print(f"Processing method: {method}")
            metrics_dict[method] = process_method(config, robot_desc, method, method_config)
    else:
        # Load metrics dict from file
        with open(config.metrics_dict_path, 'r') as f:
            metrics_dict = json.load(f)

    # make plots
    plot_annecs_metrics(metrics_dict, config)

    # significance testing
    significance_testing()

if __name__ == "__main__":
    main()
