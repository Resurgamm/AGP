import os
import json
import math
import time
import asyncio
from typing import Union,Literal,Optional,Iterator,List,Any,Dict
from tqdm import tqdm
import copy
import torch

from AGP.graph.graph import Graph
from experiments.accuracy import Accuracy
from AGP.utils.globals import Cost, PromptTokens, CompletionTokens

async def evaluate(
        graph:Graph,
        dataset,
        num_rounds:int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        ) -> float:

    print(f"Evaluating AGP on {dataset.__class__.__name__} split {dataset.split}")
    
    checkpoint = torch.load("model.pth")
    graph.gcn.load_state_dict(checkpoint['gcn'])
    graph.mlp.load_state_dict(checkpoint['mlp'])

    print("Model's state_dict:")
    for param_tensor in graph.gcn.state_dict():
        print(param_tensor, "\t", graph.gcn.state_dict()[param_tensor].size())

    graph.gcn.eval()
    accuracy = Accuracy()
    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return
    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        print(80*'-')

        start_ts = time.time()
        answer_log_probs = []
        tasks = []
        
        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            input_dict = dataset.record_to_input(record)
            tasks.append(input_dict)
            print(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun_evaluate(input=input_dict,num_rounds=num_rounds)))
        raw_answers = await asyncio.gather(*answer_log_probs)
        print(f"Batch time {time.time() - start_ts:.3f}")

        for raw_answer, record in zip(raw_answers, record_batch):
            print("Raw answer:", raw_answer)
            answer = dataset.postprocess_answer(raw_answer)
            print("Postprocessed answer:", answer)
            correct_answer = dataset.record_to_target_answer(record)
            print("Correct answer:", correct_answer)
            accuracy.update(answer, correct_answer)
            accuracy.print()
                
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    accuracy.print()
    print("Done!")

    return accuracy.get()


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
