import numpy as np
from scipy import spatial


def read_file(file_path):
	with open(file_path, 'r') as file:
		return file.read()

def distances_from_embeddings(
	query_embedding,
	embeddings,
	distance_metric="cosine",
):
	"""Return the distances between a query embedding and a list of embeddings."""
	distance_metrics = {
		"cosine": spatial.distance.cosine,
		"L1": spatial.distance.cityblock,
		"L2": spatial.distance.euclidean,
		"Linf": spatial.distance.chebyshev,
	}
	distances = [
		distance_metrics[distance_metric](query_embedding, embedding)
		for embedding in embeddings
	]
	return distances

def get_openai_embeddings(texts):
	from openai import OpenAI

	client = OpenAI()
	assert len(texts) <= 2048, "The batch size should not be larger than 2048."
	# replace newlines, which can negatively affect performance.
	texts = [text.replace("\n", " ") for text in texts]
	data = client.embeddings.create(input=texts, model='text-embedding-3-small').data
	return [d.embedding for d in data]

def get_codet5_embeddings(texts):
	import torch
	from transformers import AutoModel, AutoTokenizer

	checkpoint = "Salesforce/codet5p-110m-embedding"
	device = "cuda" if torch.cuda.is_available() else "cpu"
	tokenizer = AutoTokenizer.from_pretrained(checkpoint)
	model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
	embeddings = []
	for text in texts:
		inputs = tokenizer.encode(text, return_tensors="pt").to(device)
		embedding = model(inputs)[0].detach().cpu().numpy()
		embeddings.append(embedding)
	return embeddings

def get_mxbai_embeddings(texts):
	from sentence_transformers import SentenceTransformer

	model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
	embeddings = model.encode(texts)
	return embeddings

def get_bert_embeddings(texts):
	from transformers import BertTokenizer, BertModel

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained('bert-base-uncased')
	inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
	outputs = model(**inputs)
	# Use the average of the last hidden state as the sentence embedding
	embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
	return embeddings

def get_nomic_embeddings(texts):
	# import torch.nn.functional as F
	from sentence_transformers import SentenceTransformer

	model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
	texts = [f'search_document: {xs}' for xs in texts]
	embeddings = model.encode(texts)
	return embeddings

def get_mistral_embeddings(texts):
	import torch
	import torch.nn.functional as F
	from torch import Tensor
	from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

	def last_token_pool(last_hidden_states: Tensor,
					attention_mask: Tensor) -> Tensor:
		left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
		if left_padding:
			return last_hidden_states[:, -1]
		else:
			sequence_lengths = attention_mask.sum(dim=1) - 1
			batch_size = last_hidden_states.shape[0]
			return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

	quantization_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16,
	)
	tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
	model = AutoModel.from_pretrained(
		'intfloat/e5-mistral-7b-instruct',
		torch_dtype=torch.float16,
		attn_implementation="flash_attention_2",
		device_map="cuda",
		quantization_config=quantization_config,
	)

	batch_dict = tokenizer(texts, padding=True, return_tensors='pt')
	outputs = model(**batch_dict)
	embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
	embeddings = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
	return embeddings

def get_embeddings(codepath, embedding_method="codet5"):
	""" Get the embedding of a code snippet.

	Args:
		codepath (str): The path to the code snippet.
		embedding_method (str): The method to use for embedding the code snippet.

	Returns:
		list: The embedding of the code snippet.
	"""
	# Read the content of the code snippet
	content = read_file(codepath)

	# Map embedding methods to their respective functions
	embedding_methods = {
		"openai": get_openai_embeddings,
		"codet5": get_codet5_embeddings,
		"mxbai": get_mxbai_embeddings,
		"bert": get_bert_embeddings,
		"nomic": get_nomic_embeddings,
		"mistral": get_mistral_embeddings,
	}

	# Use the specified embedding method
	if embedding_method in embedding_methods:
		embeddings = embedding_methods[embedding_method]([content])
	else:
		raise ValueError(f"Invalid embedding method: {embedding_method}")

	return embeddings[0]

def get_similar_codepaths(chosen_codepath, other_codepaths, num_returns=5, embedding_method="codet5"):
	# TODO: reembedding the tasks everytime, can save it to a cache or smth
	""" Get codepaths that have similar content to that of the chosen codepath.

	Args:
		chosen_codepath (str): The path to the chosen code snippet.
		other_codepaths (list): List of paths to other code snippets.
		num_returns (int): Number of code snippets to return.
		embedding_method (str): The method to use for embedding the code snippets.

	Returns:
		list: Paths to the most similar code snippets.
	"""
	# Read contents of codepaths
	chosen_content = read_file(chosen_codepath)
	other_contents = [read_file(codepath) for codepath in other_codepaths]

	# Map embedding methods to their respective functions
	embedding_methods = {
		"openai": get_openai_embeddings,
		"codet5": get_codet5_embeddings,
		"mxbai": get_mxbai_embeddings,
		"bert": get_bert_embeddings,
		"nomic": get_nomic_embeddings,
		"mistral": get_mistral_embeddings,
	}

	# Use the specified embedding method
	if embedding_method in embedding_methods:
		embeddings = embedding_methods[embedding_method]([chosen_content] + other_contents)
	else:
		raise ValueError(f"Invalid embedding method: {embedding_method}")

	# Get the chosen vector and other vectors
	chosen_vector = embeddings[0]
	other_vectors = embeddings[1:]

	# Calculate distances between emebddings
	similarities = distances_from_embeddings(chosen_vector, other_vectors, distance_metric="cosine")
	sorted_indices = np.array(similarities).argsort()

	# Return the most similar codepaths
	similar_indices = sorted_indices[:num_returns]
	return [other_codepaths[i] for i in similar_indices], similar_indices


if __name__ == "__main__":
	chosen_codepath = "/workspace/src/omni_epic/envs/ant/cross_bridge.py"
	other_codepaths = [
			"/workspace/src/omni_epic/envs/ant/cross_bridge.py",
			"/workspace/src/omni_epic/envs/ant/go_to_box.py",
			"/workspace/src/omni_epic/envs/ant/kick_ball.py",
			"/workspace/src/omni_epic/envs/ant/maze.py",
			"/workspace/src/omni_epic/envs/ant/go_forward.py",
			"/workspace/src/omni_epic/envs/ant/walk_on_cylinder.py",
			"/workspace/src/omni_epic/envs/ant/go_down_stairs.py",
			"/workspace/src/omni_epic/envs/ant/cross_lava.py",
			"/workspace/src/omni_epic/envs/ant/balance_board.py",
		]
	similar_codepaths, similar_indices = get_similar_codepaths(chosen_codepath, other_codepaths, embedding_method="mistral")
	print(similar_codepaths)
