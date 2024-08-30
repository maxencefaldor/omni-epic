import json
import os
import re
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import networkx as nx
import argparse
import numpy as np
from pyvis.network import Network
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import base64
from scipy.spatial.distance import euclidean
from textwrap import dedent

from embodied.envs.pybullet import PyBullet
from run_utils import encode_image
from rag_utils import get_openai_embeddings, read_file


def read_last_json_entry(filepath):
	"""Reads the last JSON entry from a JSONL file."""
	with open(filepath, 'r') as f:
		content = f.read()
		json_str = re.split('(?<=})\n(?={)', content)[-1]
		json_obj = json.loads(json_str)
	return json_obj

def get_from_paths(metadata_path):
	"""Reads the metadata.json file and retrieves the from paths."""
	try:
		with open(metadata_path, 'r') as file:
			metadata = json.load(file)
			from_paths = []
			from_paths = metadata.get('taskgen_example_paths', [])
			# from_paths += metadata.get('taskgen_failed_paths', [])
			from_paths += metadata.get('taskit_from_paths', [])
			return from_paths
	except FileNotFoundError:
		return []

def extract_task_id(path):
	"""Extracts a task identifier from the directory name of the path."""
	return os.path.basename(os.path.dirname(path))

def adjust_positions(pos_dict, min_dist=10):
	"""Adjust positions to ensure a minimum distance between nodes."""
	from scipy.spatial.distance import cdist
	positions = np.array(list(pos_dict.values()))
	dist_matrix = cdist(positions, positions)  # Compute all-pair Euclidean distances

	for i, pos1 in enumerate(positions):
		for j, pos2 in enumerate(positions):
			if i != j and dist_matrix[i, j] < min_dist:
				# Nodes are too close and need to be adjusted
				direction = pos2 - pos1
				norm = np.linalg.norm(direction)
				if norm == 0:
					direction = np.random.randn(2)  # Random direction if exactly the same
					norm = np.linalg.norm(direction)
				shift = (min_dist - norm) / norm * direction
				positions[j] += shift * 0.5  # Move both nodes away from each other
				positions[i] -= shift * 0.5
	return {key: (pos[0], pos[1]) for key, pos in zip(pos_dict.keys(), positions)}

def create_graph(data, num_task_examples=0):
	"""Creates a NetworkX graph from the data dictionary."""
	G = nx.DiGraph()
	G_matchfolder = nx.DiGraph()  # graph that matches the folder name
	codepaths = data.get('codepaths', [])
	failedint = data.get('failedint', [])
	failedtrain = data.get('failedtrain', [])
	example_counter = 0
	task_counter = 0
	all_paths = codepaths + failedint + failedtrain

	# Colormap for the tasks
	colormap = plt.get_cmap('viridis').reversed()

	# Node images based on whether the task succeeded, failed, is boring
	node_images = [
		f"data:image/png;base64,{encode_image(f'./analysis/icons/{xs}.png')}"
		for xs in ['tick', 'sleep', 'cross']
	]
	path_images = {path: node_images[0] for path in codepaths}
	path_images.update({path: node_images[1] for path in failedint})
	path_images.update({path: node_images[2] for path in failedtrain})

	# Task embeddings and t-SNE
	task_embeddings = get_openai_embeddings([read_file(path) for path in all_paths])  # NOTE: this should be the same embedding model used during training
	task_embeddings = np.array(task_embeddings)
	perplexity = max(min(len(all_paths) // 5, 100), 5)
	tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
	task_embeddings = tsne.fit_transform(task_embeddings)
	scaler = MinMaxScaler(feature_range=(0, 800))
	task_embeddings = scaler.fit_transform(task_embeddings)
	task_embeddings = {path: emb for path, emb in zip(all_paths, task_embeddings)}

	# Sort the all_paths based on task number
	all_paths = sorted(all_paths, key=lambda x: int(extract_task_id(x).split('_')[1]) if 'task' in extract_task_id(x) else 0)

	for target_path in all_paths:
		# Get task description from the docstring
		try:
			env = PyBullet(env_path=target_path, vision=False)._env
			node_desc = dedent(env.__doc__).strip()
		except:
			node_desc = "Unknown"

		task_id = extract_task_id(target_path)

		if 'task' not in task_id or example_counter < num_task_examples:
			example_counter += 1
			task_id = f"Seed {example_counter}"
			task_id_matchfolder = task_id
			node_color = 'grey'
		else:
			task_counter += 1
			task_number = task_counter
			task_number_matchfolder = int(task_id.split('_')[1])
			rgba_color = colormap(task_number / (len(all_paths) - example_counter))
			node_color = mcolors.to_hex(rgba_color)
			task_id = f"Task {task_number}"
			task_id_matchfolder = f"Task {task_number_matchfolder}"

		G.add_node(
			target_path,
			label=f"{task_id}",
			color=node_color,
			title=node_desc,
			shape='circularImage',
			image=path_images[target_path],
			pos=(float(task_embeddings[target_path][0]),float(task_embeddings[target_path][1])),
		)
		G_matchfolder.add_node(
			target_path,
			label=f"{task_id_matchfolder}",
			color=node_color,
			title=node_desc,
			shape='circularImage',
			image=path_images[target_path],
			pos=(float(task_embeddings[target_path][0]),float(task_embeddings[target_path][1])),
		)
		metadata_path = os.path.join(os.path.dirname(target_path), 'metadata.json')
		from_paths = get_from_paths(metadata_path)

		for from_path in from_paths:
			if from_path in all_paths:
				G.add_edge(from_path, target_path)
				G_matchfolder.add_edge(from_path, target_path)

	return G, G_matchfolder

def create_colorbar(colormap, num_tasks, output_dir, suffix=''):
	"""Create a colorbar image with the given colormap and encode it to a data URL."""
	fig, ax = plt.subplots(figsize=(6, 1))
	fig.subplots_adjust(bottom=0.5)

	# Normalize the color range from 1 to num_tasks
	norm = mcolors.Normalize(vmin=1, vmax=num_tasks)
	cbar = ColorbarBase(ax, cmap=colormap, norm=norm, orientation='horizontal')

	# Set the ticks and labels
	ticks = np.arange(0, num_tasks, max(num_tasks // 50, 1) * 5)
	ticks[0] += 1
	# ticks = np.append(ticks, num_tasks)
	cbar.set_ticks(ticks)
	cbar.set_ticklabels([str(i) for i in ticks])
	cbar.set_label('Generation Number')

	# Save the colorbar to a temporary buffer instead of a file
	from io import BytesIO
	buffer = BytesIO()
	plt.savefig(buffer, format='png', bbox_inches='tight')
	plt.savefig(os.path.join(output_dir, f'colorbar_{suffix}.svg'), format='svg', bbox_inches='tight')  # Save the colorbar to a separate SVG file
	plt.close()
	buffer.seek(0)
	image_png = buffer.getvalue()
	buffer.close()

	# Convert PNG image to data URL
	image_base64 = base64.b64encode(image_png).decode('utf-8')
	return f"data:image/png;base64,{image_base64}"

def visualize_graph(G, output_dir, include_all_edges=True, suffix='', more_points=False):
	"""Converts a NetworkX graph to a pyvis network and saves it as an HTML file."""
	nt = Network("800px", "100%", notebook=True, directed=True, cdn_resources='remote')

	# Disable the physics to ensure nodes stay in given positions
	nt.options.physics.enabled = False

	# Adjust node positions
	pos_dict = {node: (attr['pos'][0], attr['pos'][1]) for node, attr in G.nodes(data=True)}
	for _ in range(10):
		pos_dict = adjust_positions(pos_dict, min_dist=25 if more_points else 60)

	# Set nodes from the NetworkX graph
	max_node_num = 0
	for node, attr in G.nodes(data=True):
		max_node_num = max(int(attr['label'].split(' ')[1]), max_node_num)
		x, y = pos_dict[node]
		nt.add_node(
			node,
			label=attr['label'],
			# label=attr['label'].split(' ')[-1],
			color=attr['color'],
			title=attr['title'],
			x=x, y=y,
			shape='circularImage',
			image=attr['image'],
			size=10 if more_points else 20,
		)

	# Determine closest parent for edge addition
	closest_parents = {}
	for child in G.nodes():
		min_distance = float('inf')
		closest_parent = None
		child_pos = pos_dict[child]
		for parent in G.predecessors(child):
			parent_pos = pos_dict[parent]
			dist = euclidean(child_pos, parent_pos)
			if dist < min_distance:
				min_distance = dist
				closest_parent = parent
		closest_parents[child] = closest_parent

	# Add edges
	for edge in G.edges():
		if include_all_edges:
			nt.add_edge(edge[0], edge[1])
		elif edge[1] in closest_parents and edge[0] == closest_parents[edge[1]]:
			nt.add_edge(edge[0], edge[1])

	# Save the network to an HTML file
	suffix = suffix + ('all_edges' if include_all_edges else 'closest_edges')
	output_path = os.path.join(output_dir, f'archive_viz_{suffix}.html')
	nt.show(output_path)

	# Add colorbar to the HTML
	colorbar_data_url = create_colorbar(plt.get_cmap('viridis').reversed(), max_node_num, output_dir, suffix=suffix)
	with open(output_path, 'a') as f:
		f.write(f'<img src="{colorbar_data_url}" style="position:absolute; top:20; right:20; width:400px;">')

def main():
	parser = argparse.ArgumentParser(description='Process the path to the archive.jsonl file.')
	parser.add_argument('--path', type=str, help='The path to the archive.jsonl file')
	parser.add_argument('--num-task-examples', '-e', type=int, default=0, help='The number of tasks that are actually examples')
	parser.add_argument('--more-points', '-m', action='store_true', help='There are more points in the visualization')
	args = parser.parse_args()

	data = read_last_json_entry(args.path)
	output_dir = os.path.dirname(args.path)
	G, G_matchfolder = create_graph(data, num_task_examples=args.num_task_examples)
	visualize_graph(G, output_dir, include_all_edges=True, more_points=args.more_points)  # Visualize with all edges
	visualize_graph(G, output_dir, include_all_edges=False, more_points=args.more_points)  # Visualize only with closest parent edges
	visualize_graph(G_matchfolder, output_dir, include_all_edges=True, suffix='matchfolder_', more_points=args.more_points)  # Visualize with all edges (match folder)
	visualize_graph(G_matchfolder, output_dir, include_all_edges=False, suffix='matchfolder_', more_points=args.more_points)


if __name__ == "__main__":
	main()
