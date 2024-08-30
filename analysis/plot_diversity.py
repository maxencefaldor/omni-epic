import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from sklearn.decomposition import PCA
from scipy import stats

from analysis.visualize_taskgen import read_last_json_entry
from rag_utils import get_embeddings


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(model, data, epochs=100, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}')

def calculate_cell_coverage(codepaths, codepaths_2d_embedding, min_x, max_x, min_y, max_y, grid_size=10):
    embeddings = np.array([codepaths_2d_embedding[codepath] for codepath in codepaths])

    # Create a grid_size x grid_size grid
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # Calculate bin edges
    x_bins = np.linspace(min_x, max_x, grid_size + 1)
    y_bins = np.linspace(min_y, max_y, grid_size + 1)

    # Discretize the embeddings into the grid
    for x, y in embeddings:
        x_index = np.digitize(x, x_bins) - 1
        y_index = np.digitize(y, y_bins) - 1

        # Ensure the indices are within bounds (0 to grid_size-1)
        x_index = min(max(x_index, 0), grid_size - 1)
        y_index = min(max(y_index, 0), grid_size - 1)

        grid[y_index, x_index] += 1  # Increment the cell count

    # Calculate cell coverage
    cell_coverage = (grid > 0).sum() / (grid_size * grid_size)

    return cell_coverage

def plot_archive_diversity(method, codepaths, codepaths_2d_embedding, min_x, max_x, min_y, max_y, grid_size=10, suffix='', file_format='png', remove_titles=False, remove_background=False, remove_axes_labels=False, add_colorbar=True):
    embeddings = np.array([codepaths_2d_embedding[codepath] for codepath in codepaths])
    
    # Create a grid_size x grid_size grid
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # Calculate bin edges
    x_bins = np.linspace(min_x, max_x, grid_size + 1)
    y_bins = np.linspace(min_y, max_y, grid_size + 1)
    
    # Discretize the embeddings into the grid
    for x, y in embeddings:
        x_index = np.digitize(x, x_bins) - 1
        y_index = np.digitize(y, y_bins) - 1
        
        # Ensure the indices are within bounds (0 to grid_size-1)
        x_index = min(max(x_index, 0), grid_size - 1)
        y_index = min(max(y_index, 0), grid_size - 1)
        
        grid[y_index, x_index] += 1  # Increment the cell count

    # Create a discrete colormap
    color_palette = plt.cm.inferno
    colors = color_palette(np.linspace(0, 1, 12))
    colors = colors[1:][::-1]  # Reverse the colors, skip black
    colors[0] = [1, 1, 1, 1]  # Set the color of the 0 count cell to white
    cmap = ListedColormap(colors)

    # Create discrete norm
    bounds = np.linspace(0, 10, 11)
    norm = BoundaryNorm(bounds, cmap.N)

    # Plot the grid
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest', extent=[min_x, max_x, min_y, max_y])

    # Fix aspect ratio
    ax.set_aspect('auto')

    # Add discrete colorbar if required
    if add_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Number of tasks', ticks=np.arange(11))
        cbar.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+'])
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Number of tasks', fontsize=14)

    # Add grid lines for all cells
    ax.set_xticks(np.linspace(min_x, max_x, grid_size + 1), minor=True)
    ax.set_yticks(np.linspace(min_y, max_y, grid_size + 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)

    # Set tick labels to appear only 10 times, spaced evenly
    x_ticks = np.linspace(min_x, max_x, 10)
    y_ticks = np.linspace(min_y, max_y, 10)
    ax.set_xticks(x_ticks, minor=False)
    ax.set_yticks(y_ticks, minor=False)
    ax.set_xticklabels([f'{x:.3f}' for x in x_ticks], fontsize=12)
    ax.set_yticklabels([f'{y:.3f}' for y in y_ticks], fontsize=12)

    if not remove_titles:
        ax.set_title(f'Diversity plot for {method}', fontsize=16)
    if not remove_axes_labels:
        ax.set_xlabel('Dimension 1', fontsize=14)
        ax.set_ylabel('Dimension 2', fontsize=14)
    
    method = method.replace('/', '')  # remove '/' from method name
    plt.tight_layout()
    plt.savefig(f'./plot_diversity_{method}_{grid_size}x{grid_size}{suffix}.{file_format}', dpi=300, bbox_inches='tight', transparent=remove_background)
    plt.close()

def plot_coverage(method_cell_coverage, grid_size=10, file_format='png', remove_titles=False, remove_background=False, remove_axes_labels=False, remove_x_labels=True):
    # Create a boxplot of the cell coverage for each method
    plt.figure(figsize=(10, 6))
    methods = list(method_cell_coverage.keys())
    coverages = [method_cell_coverage[method] for method in methods]

    plt.boxplot(coverages, tick_labels=methods, bootstrap=10000)
    if not remove_titles:
        plt.title('Cell Coverage Distribution by Method')
    if not remove_axes_labels:
        plt.ylabel('Cell Coverage')
        plt.xlabel('Method')
    if remove_x_labels:
        plt.gca().xaxis.set_ticklabels(['' for _ in methods])
    plt.tight_layout()
    plt.savefig(f'plot_coverage_{grid_size}.{file_format}', dpi=300, bbox_inches='tight', transparent=remove_background)
    plt.close()

    # Save the cell coverage data
    with open(f'plot_coverage_{grid_size}.json', 'w') as f:
        json.dump(method_cell_coverage, f, indent=4)

def perform_significance_testing(method_cell_coverage, grid_size=10):
    methods = list(method_cell_coverage.keys())
    coverages = [method_cell_coverage[method] for method in methods]

    results = {
        "Kruskal-Wallis H-test": {},
        "Mann-Whitney U tests": []
    }

    # Kruskal-Wallis H-test (non-parametric ANOVA)
    h_stat, p_val = stats.kruskal(*coverages)
    results["Kruskal-Wallis H-test"] = {
        "H-statistic": h_stat,
        "p-value": p_val
    }

    # Pairwise comparisons using Mann-Whitney U test
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            u_stat, p_val = stats.mannwhitneyu(coverages[i], coverages[j])
            results["Mann-Whitney U tests"].append({
                "methods": f"{methods[i]} vs {methods[j]}",
                "U-statistic": u_stat,
                "p-value": p_val
            })

    # Save the results to a JSON file
    with open(f'pvalues_coverage_{grid_size}.json', 'w') as f:
        json.dump(results, f, indent=4)

@hydra.main(version_base=None, config_path="../configs/", config_name="plot_diversity")
def main(config: DictConfig):
    data = {}
    # Read the last entry of each archive
    for method, method_config in config.methods.items():
        archives = []
        for path in method_config.paths:
            archives.append(read_last_json_entry(path))
        data[method] = archives

    # Check if embeddings already exist
    if os.path.exists('./embeddings/embeddings.npy') and os.path.exists('./embeddings/codepaths.json'):
        print("Loading existing embeddings...")
        embeddings = np.load('./embeddings/embeddings.npy')
        with open('./embeddings/codepaths.json', 'r') as f:
            codepaths = json.load(f)
        codepaths_embedding = dict(zip(codepaths, embeddings))
    else:
        print("Generating new embeddings...")
        # Get all the codepaths in all archives
        codepaths_embedding = {}
        for method, archives in data.items():
            for archive in archives:
                codepaths = archive['codepaths']
                for codepath in codepaths:
                    codepaths_embedding[codepath] = get_embeddings(codepath, config.embedding_method)

        # Save original embeddings
        os.makedirs('./embeddings', exist_ok=True)
        np.save('./embeddings/embeddings.npy', np.array(list(codepaths_embedding.values())))
        with open('./embeddings/codepaths.json', 'w') as f:
            json.dump(list(codepaths_embedding.keys()), f)

    embeddings = np.array(list(codepaths_embedding.values()))

    # Check if 2D embeddings already exist
    if os.path.exists('./embeddings/embeddings_2d.npy'):
        print("Loading existing 2D embeddings...")
        embeddings_2d = np.load('./embeddings/embeddings_2d.npy')
    else:
        print(f"Generating new 2D embeddings using {config.downscale_method}...")
        # Downscale embeddings based on the selected method
        if config.downscale_method == "autoenc":
            # Train an autoencoder to downscale embeddings to 2D
            embeddings_tensor = torch.FloatTensor(embeddings)
            input_dim = embeddings.shape[1]
            encoding_dim = 2  # 2D embeddings
            model = Autoencoder(input_dim, encoding_dim)
            train_autoencoder(model, embeddings_tensor)

            # Save the trained model
            torch.save(model.state_dict(), './embeddings/autoencoder_model.pth')

            # Get 2D embeddings of all codepaths
            model.eval()
            with torch.no_grad():
                embeddings_2d, _ = model(embeddings_tensor)
            embeddings_2d = embeddings_2d.numpy()
        elif config.downscale_method == "pca":
            # Use PCA to downscale embeddings to 2D
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
        else:
            raise ValueError(f"Invalid downscale_method: {config.downscale_method}")

        # Save 2D embeddings
        np.save('./embeddings/embeddings_2d.npy', embeddings_2d)

    codepaths_2d_embedding = dict(zip(codepaths_embedding.keys(), embeddings_2d))

    # Get the min and max values of the 2D embeddings
    min_x, max_x = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    min_y, max_y = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()

    # Calculate the minimum number of codepaths across all methods
    min_n_codepaths = min(len(archive['codepaths']) for archives in data.values() for archive in archives)
    print(f"Minimum number of codepaths: {min_n_codepaths}")

    # Plot the diversity in 2D space for each method and calculate cell coverage
    grid_size = config.grid_size
    method_cell_coverage = {}
    for method, archives in data.items():
        method_coverages = []
        for i, archive in enumerate(archives):
            codepaths = archive['codepaths'][:min_n_codepaths]
            plot_archive_diversity(method, codepaths, codepaths_2d_embedding, min_x, max_x, min_y, max_y, grid_size=grid_size, suffix=f'_{i}', file_format=config.file_format, remove_titles=config.remove_titles, remove_background=config.remove_background, remove_axes_labels=config.remove_axes_labels, add_colorbar=config.add_colorbar)

            # Calculate cell coverage
            cell_coverage = calculate_cell_coverage(codepaths, codepaths_2d_embedding, min_x, max_x, min_y, max_y, grid_size=grid_size)
            method_coverages.append(cell_coverage)

        method_cell_coverage[method] = method_coverages

    # Create boxplot and save data
    plot_coverage(method_cell_coverage, grid_size=grid_size, file_format=config.file_format, remove_titles=config.remove_titles, remove_background=config.remove_background, remove_axes_labels=config.remove_axes_labels, remove_x_labels=config.remove_x_labels)

    # Perform significance testing
    perform_significance_testing(method_cell_coverage, grid_size=grid_size)

if __name__ == "__main__":
    main()
