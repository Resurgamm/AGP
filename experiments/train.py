import torch
from typing import Iterator
import pandas as pd
import numpy as np
import asyncio
import copy
import json
import re

from AGP.graph.graph import Graph

async def train(graph:Graph,
			num_iters:int=100,
			num_rounds:int=1,
			lr:float=0.001,
			batch_size:int = 4,
			resume:bool = False,
			file_path:str = ""
		  ) -> None:
	
	def read_multiple_json_objects(filepath):
		with open(filepath, 'r', encoding='utf-8') as f:
			text = f.read()
			
		json_strings = re.findall(r'\{.*?\}', text, re.DOTALL)
		
		data = []
		for js in json_strings:
			try:
				obj = json.loads(js)
				data.append(obj)
			except json.JSONDecodeError as e:
				print(f"error: {e}, with content{js}")
		return data

	dataset = read_multiple_json_objects(file_path)

	def infinite_data_loader() -> Iterator[pd.DataFrame]:
			perm = np.random.permutation(len(dataset))
			while True:
				for idx in perm:
					record = dataset[idx.item()]
					yield record
	
	loader = infinite_data_loader()
	
	if not resume:
		optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=lr)    
	else:
		checkpoint = torch.load("model.pth")
		graph.gcn.load_state_dict(checkpoint['gcn'])
		graph.mlp.load_state_dict(checkpoint['mlp'])
		optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=lr)
		optimizer.load_state_dict(checkpoint['optimizer'])

	graph.gcn.train()

	for i_iter in range(num_iters):
		print(f"Iter {i_iter}", 80*'-')
		answer_log_probs = []

		for i_record, record in zip(range(batch_size), loader):
			realized_graph = copy.deepcopy(graph)
			realized_graph.gcn = graph.gcn
			realized_graph.mlp = graph.mlp
			input_dict = record
			print(input_dict)
			answer_log_probs.append(asyncio.create_task(realized_graph.arun_train(input=input_dict,num_rounds=num_rounds)))

		loss_list = await asyncio.gather(*answer_log_probs)
		loss = torch.mean(torch.stack(loss_list))

		optimizer.zero_grad() 
		loss.backward()
		optimizer.step()
	
		print("Graph gcn parameters:")
		for name, parms in graph.gcn.named_parameters():
			print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))

		print("Graph mlp parameters:")
		for name, parms in graph.mlp.named_parameters():
			print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))

		torch.save(
			{
				'gcn': graph.gcn.state_dict(),
				'mlp': graph.mlp.state_dict(),
				'optimizer': optimizer.state_dict(),
			},
			"model.pth",
		)
		

		