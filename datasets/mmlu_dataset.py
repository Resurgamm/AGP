import glob
import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC
import random

class MMLUDataset(ABC):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"datasets/MMLU/data/{self._split}/"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        print("Number of topics: ", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None,
                            names=names,encoding='utf-8')
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if answer == "":    
            print("Answer not found, using random answer.")
            answer = random.choice(['A', 'B', 'C', 'D'])
        if not isinstance(answer, str):
            print("Answer not found, using random answer.")
            answer = random.choice(['A', 'B', 'C', 'D'])
            # raise Exception("Expected string")
        if len(answer) > 0:
            ans_pos = answer.find("answer is")
            if ans_pos != -1:
                answer = answer[ans_pos+len("answer is"):].strip(":").strip().strip("Option").strip()
            answer = answer[0] # Try to format the answer by taking the first letter
            if answer not in ['A', 'B', 'C', 'D']:
                ans_pos = answer.find("option is")
                if ans_pos != -1:
                    answer = answer[ans_pos+len("option is"):].strip(":").strip().strip("Option").strip()
                answer = answer[0] # Try to format the answer by taking the first letter
            if answer not in ['A', 'B', 'C', 'D']:
                ans_pos = answer.find("Option is")
                if ans_pos != -1:
                    answer = answer[ans_pos+len("Option is"):].strip(":").strip().strip("Option").strip()
                answer = answer[0] # Try to format the answer by taking the first letter
            if answer not in ['A', 'B', 'C', 'D']:
                ans_pos = answer.find("Option")
                if ans_pos != -1:
                    answer = answer[ans_pos+len("Option"):].strip(":").strip()
                answer = answer[0] # Try to format the answer by taking the first letter
            if answer not in ['A', 'B', 'C', 'D']:
                ans_pos = answer.find("Option")
                if ans_pos != -1:
                    answer = answer[ans_pos+len("Option"):].strip()
                answer = answer[0] # Try to format the answer by taking the first letter
            if answer not in ['A', 'B', 'C', 'D']:
                ans_pos = answer.find("Answer")
                if ans_pos != -1:
                    answer = answer[ans_pos+len("Answer"):].strip(":").strip()
                answer = answer[0] # Try to format the answer by taking the first letter
            if answer not in ['A', 'B', 'C', 'D']:
                ans_pos = answer.find("Answer")
                if ans_pos != -1:
                    answer = answer[ans_pos+len("Answer"):].strip()
                answer = answer[0] # Try to format the answer by taking the first letter
            if answer not in ['A', 'B', 'C', 'D']:
                choices = ["A", "B", "C", "D"]
                for word in answer.split():
                    if word in choices:
                        answer = word
            if answer not in ['A', 'B', 'C', 'D']:
                print("Answer not found, using random answer.")
                answer = random.choice(['A', 'B', 'C', 'D'])
        print(f"Final answer is: {answer}")
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
