"""
The logic in this file largely borrows from Qwen2.5-Math codebase at https://github.com/QwenLM/Qwen2.5-Math:
"""

from eval.answer_extraction import parse_mcq_single


def get_multiple_choice_answer(pred: str):
    return parse_mcq_single(pred, letters="ABCD", fallback_window=1_000_000)
