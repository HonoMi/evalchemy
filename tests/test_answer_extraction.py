from eval.answer_extraction import (
    extract_final_answer_segment,
    normalize_generation_text,
    parse_boxed_scalar,
    parse_code_block,
    parse_mcq_single,
    parse_numeric,
)


def test_normalize_generation_text_extracts_reasoning_final_segment():
    text = "analysis details assistant final Answer: A"
    assert normalize_generation_text(text) == "Answer: A"


def test_extract_final_answer_segment_handles_think_suffix():
    text = "<think>work</think> Final Answer: Answer: B"
    assert extract_final_answer_segment(text) == "Answer: B"


def test_parse_mcq_single_prefers_last_answer_mention():
    text = "Question 1 the answer is (F)\n\nQuestion 2 the answer is (D)"
    assert parse_mcq_single(text, letters="ABCDEFGHIJ") == "D"


def test_parse_mcq_single_handles_boxed_text():
    assert parse_mcq_single(r"\boxed{\text{C}}", letters="ABCD") == "C"


def test_parse_numeric_handles_number_words():
    text = "**Answer:** Each of Alice's brothers has two sisters."
    assert parse_numeric(text) == "2"


def test_parse_numeric_handles_plain_number():
    assert parse_numeric("6") == "6"


def test_parse_boxed_scalar_returns_last_boxed_value():
    assert parse_boxed_scalar(r"first \boxed{12}; final \boxed{34}") == "34"


def test_parse_code_block_returns_last_normalized_block():
    text = "<think>draft</think>```python\nprint(1)\n```\n```python\nprint(2)\n```"
    assert parse_code_block(text, language="python") == "print(2)"
