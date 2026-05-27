import re
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.answer_extraction import iter_code_blocks, normalize_generation_text


def has_code(response):
    response = normalize_generation_text(response)
    matches = list(iter_code_blocks(response))

    if not matches:
        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        # CruxEval
        pattern = r"\[ANSWER\](.*?)\[/ANSWER\]"
        matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        try:
            answer = remove_boxed(last_boxed_only_string(response))
            matches = [answer]
        except:
            pass

    return matches
