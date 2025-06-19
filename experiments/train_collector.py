import torch
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy

from AGP.graph.graph import Graph
from experiments.accuracy import Accuracy
from AGP.utils.globals import Cost, PromptTokens, CompletionTokens

async def train(graph:Graph,
			dataset,
			num_iters:int=100,
			num_rounds:int=1,
			lr:float=0.001,
			batch_size:int = 4,
			resume:bool = False,
		  ) -> None:
	
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

	last_accuracy = 0.0
	retraining_counter = 0

	i_iter = 0
	while i_iter < num_iters:
		print(f"Iter {i_iter}", 80*'-')
		start_ts = time.time()
		correct_answers = []
		answer_log_probs = []

		optimizer.zero_grad() 

		for i_record, record in zip(range(batch_size), loader):
			realized_graph = copy.deepcopy(graph)
			realized_graph.gcn = graph.gcn
			realized_graph.mlp = graph.mlp
			input_dict = dataset.record_to_input(record)
			print(input_dict)
			answer_log_probs.append(asyncio.create_task(realized_graph.arun_collecter(input=input_dict,num_rounds=num_rounds,retraining_count=retraining_counter)))
			correct_answer = dataset.record_to_target_answer(record)
			correct_answers.append(correct_answer)
		
		raw_results = await asyncio.gather(*answer_log_probs)
		raw_answers, log_probs, edge_weight, mask = zip(*raw_results)
		loss_list: List[torch.Tensor] = []
		utilities: List[float] = []
		answers: List[str] = []
		
		for raw_answer, log_prob, correct_answer in zip(raw_answers, log_probs, correct_answers):
			print(f"raw_answer:{raw_answer}")
			answer = dataset.postprocess_answer(raw_answer)
			answers.append(answer)
			assert isinstance(correct_answer, str), \
					f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
			accuracy = Accuracy()
			accuracy.update(answer, correct_answer)
			utility = accuracy.get()
			utilities.append(utility)
			print("log_prob:", log_prob, "utility:", utility)	
			single_loss = - log_prob * utility
			print(f"single_loss:{single_loss}\n")
			loss_list.append(single_loss)
			print(f"correct answer:{correct_answer}")

		now_accuracy = np.mean(utilities)

		print(f"now_accuracy:{now_accuracy} vs last_accuracy:{last_accuracy}")

		if now_accuracy >= max(last_accuracy - 0.2 - retraining_counter * 0.01, 0.85) or retraining_counter > 3:

			retraining_counter = 0

			last_accuracy = now_accuracy
			print("Updating model")

			total_loss = torch.mean(torch.stack(loss_list))
			total_loss = total_loss * 100
			total_loss.backward()

			print("Graph gcn parameters:")
			for name, parms in graph.gcn.named_parameters():
				print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))

			print("Graph mlp parameters:")
			for name, parms in graph.mlp.named_parameters():
				print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))

			with open("grads.txt", "a") as f:
				f.write("Graph gcn parameters:\n")
				for name, parms in graph.gcn.named_parameters():
					f.write(f'-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)}\n')
				f.write("Graph mlp parameters:\n")
				for name, parms in graph.mlp.named_parameters():
					f.write(f'-->name: {name} -->grad_requirs: {parms.requires_grad} --weight {torch.mean(parms.data)} -->grad_value: {torch.mean(parms.grad)}\n')

			optimizer.step()

			print("raw_answers:",raw_answers)
			print("answers:",answers)
			print(f"Batch time {time.time() - start_ts:.3f}")
			print("utilities:", utilities) # [0.0, 0.0, 0.0, 1.0]
			print("loss:", total_loss.item()) # 4.6237263679504395
			print(f"Cost {Cost.instance().value}")
			print(f"PromptTokens {PromptTokens.instance().value}")
			print(f"CompletionTokens {CompletionTokens.instance().value}")

			torch.save(
				{
					'gcn': graph.gcn.state_dict(),
					'mlp': graph.mlp.state_dict(),
					'optimizer': optimizer.state_dict(),
				},
				"model.pth",
			)

		elif now_accuracy >= 0.8:
			print("Not updating model, retraining")
			retraining_counter += 1
			i_iter -= 1

		i_iter += 1

		