import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


PROMPT = """次の四択問題に答えてください。各選択肢を検討し、最後の答えだけを \\boxed{{A}} のように選択肢の記号1文字で示してください。

問題:
{question}

選択肢:
{choices}

答え:
"""

_FULLWIDTH_TO_ASCII = str.maketrans(
    {
        "Ａ": "A",
        "Ｂ": "B",
        "Ｃ": "C",
        "Ｄ": "D",
        "ａ": "A",
        "ｂ": "B",
        "ｃ": "C",
        "ｄ": "D",
    }
)


class DF_MCQ_Imaichi_2025Benchmark(BaseBenchmark):
    """
    Local MCQ benchmark backed by 2025_Imaichi_MCQ.json.
    """

    def __init__(
        self,
        data_file: Optional[str] = None,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 4096,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file or str(Path(__file__).resolve().parent / "data" / "2025_Imaichi_MCQ.json")
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = min(max_tokens or 4096, 4096)
        self.n_repeat = 1

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        examples = self.load_questions()
        instances = []

        for idx, example in enumerate(examples):
            messages = [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        question=example["question"],
                        choices=self.format_choices(example["choices"]),
                    ),
                }
            ]
            templated_messages = self._prepare_messages(messages, model)

            instance = Instance(
                "generate_until",
                example,
                (
                    templated_messages,
                    {
                        "do_sample": False,
                        "temperature": 0.0,
                        "max_new_tokens": self.max_new_tokens,
                        "seed": self.seed,
                    },
                ),
                idx,
            )
            instance.metadata = {
                "problem_id": str(example["id"]),
                "expected_answer": example["answer"],
            }
            instances.append(instance)

        self.logger.info("Generating responses for DF_MCQ_Imaichi_2025...")
        outputs = self.compute(model, instances)

        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            text = self.unwrap_output(output)
            prediction = self.extract_answer(text)
            example["model_output"] = text
            example["model_answer"] = prediction
            example["correct"] = int(prediction == example["answer"])

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)
        if num_questions == 0:
            results.update(
                {
                    "num_total": 0,
                    "num_solved": 0,
                    "solved_avg": 0.0,
                    "run_stats": [],
                    "accuracy_avg": 0.0,
                    "accuracy_std_err": 0.0,
                    "overall_accuracy": 0.0,
                    "num_repeat": self.n_repeat,
                }
            )
            return results

        correct_flags = np.asarray([example["correct"] for example in examples], dtype=float)
        solved = int(correct_flags.sum())
        accuracy = float(correct_flags.mean())
        stderr = float(correct_flags.std(ddof=1) / math.sqrt(num_questions)) if num_questions > 1 else 0.0

        results.update(
            {
                "num_total": num_questions,
                "num_solved": solved,
                "solved_avg": float(solved),
                "run_stats": [
                    {
                        "repetition": 1,
                        "num_total": num_questions,
                        "num_solved": solved,
                        "accuracy": accuracy,
                    }
                ],
                "accuracy_avg": accuracy,
                "accuracy_std_err": stderr,
                "overall_accuracy": accuracy,
                "num_repeat": self.n_repeat,
            }
        )
        return results

    def load_questions(self) -> List[Dict[str, Any]]:
        with open(self.data_file, "r") as f:
            questions = json.load(f)

        if self.debug:
            questions = questions[:2]

        normalized_questions = []
        for idx, question in enumerate(questions):
            choices = [
                {
                    "option": str(choice["option"]).strip().upper(),
                    "text": str(choice["text"]).strip(),
                }
                for choice in question["choices"]
            ]
            normalized_questions.append(
                {
                    "id": idx,
                    "question": str(question["question"]).strip(),
                    "choices": choices,
                    "answer": str(question["answer"]).strip().upper(),
                    "explanation": str(question.get("explanation", "")).strip(),
                }
            )

        self.logger.info("Loaded %d questions from %s", len(normalized_questions), self.data_file)
        return normalized_questions

    @staticmethod
    def format_choices(choices: List[Dict[str, str]]) -> str:
        return "\n".join(f'{choice["option"]}. {choice["text"]}' for choice in choices)

    @staticmethod
    def unwrap_output(output: Any) -> str:
        if isinstance(output, str):
            return output
        if hasattr(output, "outputs") and output.outputs:
            return output.outputs[0].text
        if hasattr(output, "text"):
            return output.text
        return str(output)

    def extract_answer(self, output: str) -> str:
        if not output:
            return ""

        normalized_output = output.translate(_FULLWIDTH_TO_ASCII).upper()

        patterns = [
            r"\\BOXED\s*\{\s*([A-D])\s*\}",
            r"答え(?:は|:)?\s*[（(「『【\[]?\s*([A-D])\s*[）)」』】\]]?",
            r"ANSWER(?: IS|:)?\s*[（(\[]?\s*([A-D])\s*[)\]]?",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, normalized_output, flags=re.IGNORECASE)
            if matches:
                return matches[-1].upper()

        fallback_matches = re.findall(r"(?<![A-Z])([A-D])(?![A-Z])", normalized_output[-200:])
        if fallback_matches:
            return fallback_matches[-1].upper()

        return ""
