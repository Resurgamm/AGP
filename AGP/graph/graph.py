import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio

from AGP.graph.node import Node
from AGP.agents.agent_registry import AgentRegistry
from AGP.prompt.prompt_set_registry import PromptSetRegistry
from AGP.llm.profile_embedding import get_sentence_embedding
from AGP.gnn.gcn import GCN,MLP
from torch_geometric.utils import dense_to_sparse

crucial_roles = ['Knowlegable Expert','Critic']

class Graph(ABC):
	"""
	A framework for managing and executing a network of nodes using a language model.

	This class enables the creation of a graph structure for processing and analyzing data. Each node
	in the graph can perform specific operations, allowing for complex data processing workflows.
	The graph supports integration with language models, making it suitable for tasks that require
	natural language processing capabilities.

	The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
	
	Attributes:
		domain (str): The domain for which this graph is used.
		llm_name (str): The name of the llm that used for processing within the nodes.
		nodes (dict): A collection of nodes, each identified by a unique UUID.

	Methods:
		build_graph(): Method to be implemented for constructing the graph structure.
		add_node(node): Adds a new node to the graph with a unique identifier.
		run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
	"""

	def __init__(self, 
				domain: str,
				llm_name: Optional[str],
				agent_names: List[str],
				decision_method: str,
				optimized_spatial:bool = False,
				initial_spatial_probability: float = 0.5,
				fixed_spatial_masks:List[List[int]] = None,
				optimized_temporal:bool = False,
				initial_temporal_probability: float = 0.5,
				fixed_temporal_masks:List[List[int]] = None,
				node_kwargs:List[Dict] = None,
				):
		
		if fixed_spatial_masks is None:
			fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
		if fixed_temporal_masks is None:
			fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
		fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
		fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
		assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
		assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
		
		self.id:str = shortuuid.ShortUUID().random(length=4)
		self.domain:str = domain
		self.llm_name:str = llm_name
		self.agent_names:List[str] = agent_names
		self.optimized_spatial = optimized_spatial
		self.optimized_temporal = optimized_temporal
		self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
		self.nodes:Dict[str,Node] = {}
		self.potential_spatial_edges:List[List[str, str]] = []
		self.potential_temporal_edges:List[List[str,str]] = []
		self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
		
		self.init_nodes() # add nodes to the self.nodes
		self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
		
		self.prompt_set = PromptSetRegistry.get(domain)
		self.role_adj_matrix = self.construct_adj_matrix()
		self.features = self.construct_features()
		self.gcn = GCN(self.features.size(1)*2, # in_channels
				 	   16,						# hidden_channels
					   self.features.size(1),	# out_channels
					   64,						# mask_in_channels
					   0.0, 					# dropout_p
					   0.55)					# threshold
		self.mlp = MLP(384,16,16)

		init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
		self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
												 requires_grad=optimized_spatial) # trainable edge logits
		self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

		init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
		self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
												 requires_grad=optimized_temporal) # trainable edge logits
		self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
	
	def construct_adj_matrix(self):
		role_connect:List[Tuple[str,str]] = self.prompt_set.get_role_connection()
		num_nodes = self.num_nodes
		role_adj = torch.zeros((num_nodes,num_nodes))
		role_2_id = {}
		
		for edge in role_connect:
			in_role, out_role = edge
			role_2_id[in_role] = []
			role_2_id[out_role] = []
		for i, node_id in enumerate(self.nodes):
			role = self.nodes[node_id].role
			role_2_id[role].append(i)
			
		for edge in role_connect:
			in_role,out_role = edge
			in_ids = role_2_id[in_role]
			out_ids = role_2_id[out_role]
			for in_id in in_ids:
				for out_id in out_ids:
					role_adj[in_id][out_id] = 1
		
		edge_index, edge_weight = dense_to_sparse(role_adj)
		return edge_index
	
	def construct_features(self):
		features = []
		for node_id in self.nodes:
			role = self.nodes[node_id].role
			profile = self.prompt_set.get_description(role)
			feature = get_sentence_embedding(profile)
			features.append(feature)
		features = torch.tensor(np.array(features))
		return features
	
	def construct_new_features(self, query):
		query_embedding = torch.tensor(get_sentence_embedding(query))
		query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes,1))
		new_features = torch.cat((self.features,query_embedding),dim=1)
		return new_features
		
	@property
	def spatial_adj_matrix(self):
		matrix = np.zeros((len(self.nodes), len(self.nodes)))
		for i, node1_id in enumerate(self.nodes):
			for j, node2_id in enumerate(self.nodes):
				if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
					matrix[i, j] = 1
		return matrix

	@property
	def temporal_adj_matrix(self):
		matrix = np.zeros((len(self.nodes), len(self.nodes)))
		for i, node1_id in enumerate(self.nodes):
			for j, node2_id in enumerate(self.nodes):
				if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
					matrix[i, j] = 1
		return matrix

	@property
	def num_edges(self):
		num_edges = 0
		for node in self.nodes.values():
			num_edges += len(node.spatial_successors)
		return num_edges
	
	@property
	def num_nodes(self):
		return len(self.nodes)

	def find_node(self, id: str):
		if id in self.nodes.keys():
			return self.nodes[id]
		raise Exception(f"Node not found: {id} among "
						f"{[node.id for node in self.nodes.values()]}")
		
	def add_node(self, node: Node):
		node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
		while node_id in self.nodes:
			node_id = shortuuid.ShortUUID().random(length=4)
		node.id = node_id
		node.idnumber = len(self.nodes)
		self.nodes[node_id] = node
		return node
	
	def init_nodes(self):
		"""
		Creates and adds new nodes to the graph.
		"""
		for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
			if agent_name in AgentRegistry.registry:
				kwargs["domain"] = self.domain
				kwargs["llm_name"] = self.llm_name
				agent_instance = AgentRegistry.get(agent_name, **kwargs)
				self.add_node(agent_instance)
	
	def init_potential_edges(self):
		"""
		Creates and potential edges to the graph.
		"""
		for node1_id in self.nodes.keys():
			for node2_id in self.nodes.keys():
				self.potential_spatial_edges.append([node1_id,node2_id])
				self.potential_temporal_edges.append([node1_id,node2_id])

	def clear_spatial_connection(self):
		"""
		Clear all the spatial connection of the nodes in the graph.
		"""
		for node_id in self.nodes.keys():
			self.nodes[node_id].spatial_predecessors = []
			self.nodes[node_id].spatial_successors = []
		self.decision_node.spatial_predecessors = []
		self.decision_node.spatial_successors = []
	
	def clear_temporal_connection(self):
		"""
		Clear all the temporal connection of the nodes in the graph.
		"""
		for node_id in self.nodes.keys():
			self.nodes[node_id].temporal_predecessors = []
			self.nodes[node_id].temporal_successors = []

	def connect_decision_node(self):
		for node_id in self.nodes.keys():
			self.nodes[node_id].add_successor(self.decision_node)

	def clear_masked_nodes(self):
		for node_id in self.nodes.keys():
			self.nodes[node_id].masked = False

	def train_loss(self, prob, GT_mask, GT_edge_weight, temperature: float = 1.0,): # temperature must >= 1.0
		self.clear_spatial_connection()
		
		edge_weight = torch.zeros((self.num_nodes,self.num_nodes))

		for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
			out_node:Node = self.find_node(potential_connection[0])
			in_node:Node = self.find_node(potential_connection[1])
			if edge_mask == 0.0:
				continue
			elif edge_mask == 1.0 and self.optimized_spatial==False:
				if not self.check_cycle(in_node, {out_node}):
					out_node.add_successor(in_node,'spatial')
				continue
			if not self.check_cycle(in_node, {out_node}):
				edge_prob = torch.sigmoid(edge_logit / temperature)
				edge_weight[out_node.idnumber][in_node.idnumber] = edge_prob

		print (" --------------------- now calculate the loss ---------------------")

		GT_mask = torch.tensor(GT_mask, requires_grad=self.optimized_spatial)
		GT_mask = (GT_mask == 0).float()
		M_ij = GT_mask.unsqueeze(0) * GT_mask.unsqueeze(1)
		A_pred = edge_weight
		A_gt = torch.tensor(GT_edge_weight, requires_grad=self.optimized_spatial).float()
		A_gt = (A_gt != 0).float()
		pos_mask = (M_ij == 1)
		off_mask = (M_ij == 0)
		print("GT_mask",GT_mask)
		print("M_ij",M_ij)
		print("A_pred",A_pred)
		print("A_gt",A_gt)
		print("pos_mask",pos_mask)
		print("off_mask",off_mask)
		
		L_pos = torch.nn.functional.mse_loss(A_pred[pos_mask], A_gt[pos_mask], reduction='mean')
		off_values = A_pred[off_mask]
		if off_values.numel() > 0:
			L_off = (off_values * off_values).mean()
		else:
			L_off = torch.tensor(0.0, device=A_pred.device)
		lambda_off = 1.0
		L_edge = L_pos + lambda_off * L_off
		print("L_pos",L_pos)
		print("L_off",L_off)
		print("L_edge",L_edge)

		L_bce = torch.nn.functional.binary_cross_entropy(prob, GT_mask.float(), reduction='mean')
		L_sparse = prob.mean()
		L_consist = ((1 - GT_mask).float().unsqueeze(1) * A_pred.abs()).sum() / (self.num_nodes**2)
		lambda_s = 0.05
		lambda_c = 0.01
		L_node= L_bce + lambda_s * L_sparse + lambda_c * L_consist
		print("L_bce",L_bce)
		print("L_sparse",L_sparse)
		print("L_consist",L_consist)
		print("L_node",L_node)
		
		beta = 1
		loss = L_edge + beta * L_node
		print("loss", loss)

		return loss
	
	def construct_spatial_connection(self, mask, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
		self.clear_spatial_connection()

		print (" --------------------- now construct the spatial connection ---------------------")
		print ("mask", mask)

		for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
			out_node:Node = self.find_node(potential_connection[0])
			in_node:Node = self.find_node(potential_connection[1])
			if mask[out_node.idnumber] == 0 or mask[in_node.idnumber] == 0:
				continue
			if edge_mask == 0.0:
				continue
			elif edge_mask == 1.0 and self.optimized_spatial==False:
				if not self.check_cycle(in_node, {out_node}):
					out_node.add_successor(in_node,'spatial')
				continue
			if not self.check_cycle(in_node, {out_node}):
				edge_prob = torch.sigmoid(edge_logit / temperature)
				print(f"{out_node.id} to {in_node.id} edge_prob: {edge_prob}")
				if threshold:
					edge_prob = torch.tensor(1.0 if edge_prob > threshold else 0.0)
				if torch.rand(1) < edge_prob:
					out_node.add_successor(in_node,'spatial')
					print(f"Add spatial connection from {out_node.id} to {in_node.id}")
					print("edge_prob:",edge_prob)
	
	def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
		self.clear_temporal_connection()
		log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
		if round == 0:
			return torch.sum(torch.stack(log_probs))  
		for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
			out_node:Node = self.find_node(potential_connection[0])
			in_node:Node = self.find_node(potential_connection[1])
			if edge_mask == 0.0:
				continue
			elif edge_mask == 1.0 and self.optimized_temporal==False:
				if not self.check_cycle(in_node, {out_node}):
					out_node.add_successor(in_node,'temporal')
				continue
			
			edge_prob = torch.sigmoid(edge_logit / temperature)
			if threshold:
				edge_prob = torch.tensor(1.0 if edge_prob > threshold else 0.0)
			if torch.rand(1) < edge_prob:
				out_node.add_successor(in_node,'temporal')
				log_probs.append(torch.log(edge_prob))
			else:
				log_probs.append(torch.log(1 - edge_prob))
					
		return torch.sum(torch.stack(log_probs))

	def construct_spatial_connection_for_collecter(self, temperature: float = 1.0, threshold: float = None, retraining_count: int = 0,): # temperature must >= 1.0
		self.clear_spatial_connection()
		log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]

		edge_list = []

		edge_weight = np.zeros((self.num_nodes,self.num_nodes))
		
		print(self.potential_spatial_edges)
		print(self.spatial_masks)

		for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
			out_node:Node = self.find_node(potential_connection[0])
			in_node:Node = self.find_node(potential_connection[1])
			print(out_node.idnumber, in_node.idnumber)
			if edge_mask == 0.0:
				print("masked")
				continue
			elif edge_mask == 1.0 and self.optimized_spatial==False:
				if not self.check_cycle(in_node, {out_node}):
					out_node.add_successor(in_node,'spatial')
				continue
			if not self.check_cycle(in_node, {out_node}):
				print("no cycle")
				edge_prob = torch.sigmoid(edge_logit / temperature)
				edge_weight[out_node.idnumber][in_node.idnumber] = edge_prob
				print(f"{out_node.id, out_node.idnumber} to {in_node.id, in_node.idnumber} edge_prob: {edge_prob}")
				if threshold:
					edge_prob = torch.tensor(1.0 if edge_prob > threshold else 0.0)
				if torch.rand(1) < edge_prob:
					out_node.add_successor(in_node,'spatial')
					print(f"Add spatial connection from {out_node.id} to {in_node.id}")
					print("edge_prob:",edge_prob)
					edge_list.append((out_node, in_node, potential_connection, edge_prob))
					log_probs.append(torch.log(edge_prob))
				else:
					log_probs.append(torch.log(1 - edge_prob))

		node_score:Dict[str, int] = {}
		for node_id in self.nodes.keys():
			node_score[node_id] = torch.tensor(0.0)

		self.clear_masked_nodes()

		for (out_node, in_node, potential_connection, edge_prob) in edge_list:
			node_score[out_node.id] += edge_prob
			node_score[in_node.id] += edge_prob

		sorted_node_score = sorted(node_score.items(), key = lambda kv: kv[1])

		print(sorted_node_score)
		print(len(self.nodes), len(self.nodes) * 0.12)

		for node_id, score in sorted_node_score:
			print(f"Now check node {node_id}, score: {score}, role: {self.nodes[node_id].role}, is crucial: {self.nodes[node_id].role in crucial_roles}")
			if score < len(self.nodes) * 0.12 - retraining_count * 0.01 and self.nodes[node_id].role not in crucial_roles:
				print(f"Remove node {node_id}")
				self.nodes[node_id].masked = True
				
				for (out_node, in_node, potential_connection, edge_prob) in edge_list:
					if out_node.id == node_id or in_node.id == node_id:
						if in_node in out_node.spatial_successors and out_node in in_node.spatial_predecessors:
							print(f"Remove spatial connection from {out_node.id} to {in_node.id}")
							out_node.spatial_successors.remove(in_node)
							in_node.spatial_predecessors.remove(out_node)
							print(torch.log(edge_prob), torch.log(1 - edge_prob))
							log_probs.remove(torch.log(edge_prob))
							log_probs.append(torch.log(1 - edge_prob))

		for node_id in self.nodes.keys():
			if len(self.nodes[node_id].spatial_predecessors) == 0 and len(self.nodes[node_id].spatial_successors) == 0:
				self.nodes[node_id].masked = True
		
		mask = np.zeros(self.num_nodes)
		for node_id in self.nodes.keys():
			if self.nodes[node_id].masked:
				mask[self.nodes[node_id].idnumber] = 1

		return torch.sum(torch.stack(log_probs)), edge_weight, mask

	async def arun_collecter(self, input: Dict[str,str], 
				  num_rounds:int = 3, 
				  max_tries: int = 3, 
				  max_time: int = 600,
				  retraining_count: int = 0,) -> List[Any]:
		log_probs = 0
		new_features = self.construct_new_features(input['task'])
		logits, prob, mask = self.gcn(new_features,self.role_adj_matrix)
		print("logit 1", logits)
		logits = self.mlp(logits)
		print("logit 2", logits)
		self.spatial_logits = logits @ logits.t()
		print("logit 3", self.spatial_logits)
		self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

		print(f"###########logits:{self.spatial_logits}")

		for round in range(num_rounds):
			res_prob, edge_weight, mask = self.construct_spatial_connection_for_collecter(retraining_count=retraining_count)
			log_probs += res_prob
			print("log_probs", log_probs)
			
			in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
			zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0 and not self.nodes[node_id].masked]

			while zero_in_degree_queue:
				current_node_id = zero_in_degree_queue.pop(0)
				tries = 0
				while tries < max_tries:
					try:
						await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),timeout=max_time) # output is saved in the node.outputs
						break
					except Exception as e:
						print(f"Error during execution of node {current_node_id}: {e}")
					tries += 1
				for successor in self.nodes[current_node_id].spatial_successors:
					if successor.id not in self.nodes.keys():
						continue
					in_degree[successor.id] -= 1
					if in_degree[successor.id] == 0:
						zero_in_degree_queue.append(successor.id)
			
			self.update_memory()
		
		print("\n\n------------Now it's time for the decision node---------------------\n\n")
		print(self.decision_node.id)
		print(self.decision_node.role)
		
		self.connect_decision_node()
		await self.decision_node.async_execute(input)
		final_answers = self.decision_node.outputs
		if len(final_answers) == 0:
			final_answers.append("No answer of the decision node")

		
		return final_answers, log_probs, edge_weight, mask

	async def arun_train(self, 
				  input: Dict[str,str], 
				  num_rounds:int = 3,) -> List[Any]:
		
		loss = 0
		print("---------------now start training-------------------")
		print(input)
		new_features = self.construct_new_features(input['task'])
		GT_mask = input['mask']
		GT_edge_weight = input['edge_weight']
		logits, prob, mask = self.gcn(new_features,self.role_adj_matrix)
		print("prob", prob)
		print("mask", mask)
		print("logit 1", logits)
		logits = self.mlp(logits)
		print("logit 2", logits)
		self.spatial_logits = logits @ logits.t()
		print("logit 3", self.spatial_logits)
		self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

		print(f"###########logits:{self.spatial_logits}")

		for round in range(num_rounds):
			loss += self.train_loss(prob=prob,
									GT_mask=GT_mask,
									GT_edge_weight=GT_edge_weight,
									temperature=1.0)
		
		return loss
	
	async def arun_evaluate(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
		new_features = self.construct_new_features(input['task'])
		print(input['task'])
		print("new_features", new_features)
		print("role_adj_matrix", self.role_adj_matrix)
		logits, prob, mask = self.gcn(new_features,self.role_adj_matrix)
		print("prob", prob)
		print("mask", mask)
		print("logit 1", logits)
		logits = self.mlp(logits)
		print("logit 2", logits)
		self.spatial_logits = logits @ logits.t()
		print("logit 3", self.spatial_logits)
		self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

		print(f"###########logits:{self.spatial_logits}")

		for round in range(num_rounds):
			self.construct_spatial_connection(mask=mask)
			
			in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
			zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

			while zero_in_degree_queue:
				current_node_id = zero_in_degree_queue.pop(0)
				tries = 0
				while tries < max_tries:
					try:
						await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),timeout=max_time) # output is saved in the node.outputs
						break
					except Exception as e:
						print(f"Error during execution of node {current_node_id}: {e}")
					tries += 1
				for successor in self.nodes[current_node_id].spatial_successors:
					if successor.id not in self.nodes.keys():
						continue
					in_degree[successor.id] -= 1
					if in_degree[successor.id] == 0:
						zero_in_degree_queue.append(successor.id)
			
			self.update_memory()
			
		self.connect_decision_node()
		await self.decision_node.async_execute(input)
		final_answers = self.decision_node.outputs
		if len(final_answers) == 0:
			final_answers.append("No answer of the decision node")
		return final_answers
	
	def update_memory(self):
		for id,node in self.nodes.items():
			node.update_memory()
	
	def check_cycle(self, new_node, target_nodes):
		if new_node in target_nodes:
			return True
		for successor in new_node.spatial_successors:
			if self.check_cycle(successor, target_nodes):
				return True
		return False

	def update_masks(self, pruning_rate: float) -> torch.Tensor:
		if self.optimized_spatial:
			num_edges = (self.spatial_masks > 0).sum()
			num_masks = (self.spatial_masks == 0).sum()
			prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
			_edge_logits = self.spatial_logits.clone()
			min_edge_logit = _edge_logits.min()
			_edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
			sorted_edges_idx = torch.argsort(_edge_logits)
			prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
			self.spatial_masks[prune_idx] = 0
		
		if self.optimized_temporal:
			num_edges = (self.temporal_masks > 0).sum()
			num_masks = (self.temporal_masks == 0).sum()
			prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
			_edge_logits = self.temporal_logits.clone()
			min_edge_logit = _edge_logits.min()
			_edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
			sorted_edges_idx = torch.argsort(_edge_logits)
			prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
			self.temporal_masks[prune_idx] = 0
		return self.spatial_masks, self.temporal_masks

def min_max_norm(tensor:torch.Tensor):
	min_val = tensor.min()
	max_val = tensor.max()
	if min_val == max_val:
		return tensor
	normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
	normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
	return normalized_minus1_to_1
	