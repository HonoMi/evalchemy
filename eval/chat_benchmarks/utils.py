import re
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed


def has_code(response):
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    # Use re.DOTALL to match multiline content inside backticks
    matches = re.findall(pattern, response, re.DOTALL)

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
