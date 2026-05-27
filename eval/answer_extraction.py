import logging
import re
from typing import Any, Iterable, Optional, Sequence


logger = logging.getLogger(__name__)

_HARMONY_FINAL_PREFIX_RE = re.compile(
    r"^\s*(?:assistant\s+)?final\b(?!\s+answer\b)[:\s]*",
    re.IGNORECASE,
)
_HARMONY_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
    re.IGNORECASE | re.DOTALL,
)
_ASSISTANT_FINAL_RE = re.compile(r"assistant\s+final\b[:\s]*(.*)$", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"final\s+answer\b[:\s]*(.*)$", re.IGNORECASE | re.DOTALL)
_THINK_SUFFIX_RE = re.compile(r"</think>\s*(.*)$", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*((?:[^{}]|\{[^{}]*\})*)\s*\}", re.IGNORECASE | re.DOTALL)
_LATEX_TEXT_RE = re.compile(r"^\\text\s*\{\s*(.*?)\s*\}$", re.IGNORECASE | re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```(?:[A-Za-z0-9_+#.-]+)?\s*\n?(.*?)\n?```", re.DOTALL)

_FULLWIDTH_TO_ASCII = str.maketrans(
    {
        "Ａ": "A",
        "Ｂ": "B",
        "Ｃ": "C",
        "Ｄ": "D",
        "Ｅ": "E",
        "Ｆ": "F",
        "Ｇ": "G",
        "Ｈ": "H",
        "Ｉ": "I",
        "Ｊ": "J",
        "ａ": "A",
        "ｂ": "B",
        "ｃ": "C",
        "ｄ": "D",
        "ｅ": "E",
        "ｆ": "F",
        "ｇ": "G",
        "ｈ": "H",
        "ｉ": "I",
        "ｊ": "J",
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "：": ":",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
        "【": "[",
        "】": "]",
    }
)

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def unwrap_generation_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if hasattr(output, "outputs") and output.outputs:
        return output.outputs[0].text
    if hasattr(output, "text"):
        return output.text
    return str(output)


def _strip_harmony_final_prefix(text: str) -> str:
    return _HARMONY_FINAL_PREFIX_RE.sub("", text, count=1)


def _last_nonempty_capture(text: str, pattern: re.Pattern) -> Optional[str]:
    matches = list(pattern.finditer(text))
    for match in reversed(matches):
        captured = match.group(1).strip()
        if captured:
            return captured
    return None


def extract_reasoning_final_segment(text: str) -> str:
    """Return the final channel/content for common reasoning-model wrappers."""

    for pattern in (
        _HARMONY_FINAL_CHANNEL_RE,
        _ASSISTANT_FINAL_RE,
        _THINK_SUFFIX_RE,
    ):
        captured = _last_nonempty_capture(text, pattern)
        if captured is not None:
            return _strip_harmony_final_prefix(captured).strip()
    return _strip_harmony_final_prefix(text).strip()


def extract_final_answer_segment(text: str) -> str:
    text = extract_reasoning_final_segment(text)
    captured = _last_nonempty_capture(text, _FINAL_ANSWER_RE)
    if captured is not None:
        return _strip_harmony_final_prefix(captured).strip()
    return text


def normalize_generation_text(output: Any) -> str:
    return extract_reasoning_final_segment(unwrap_generation_output(output))


def supports_harmony_postprocess(tokenizer: Any) -> bool:
    return tokenizer is not None and (
        hasattr(tokenizer, "parse_response") or hasattr(tokenizer, "parse_harmony_message")
    )


def extract_harmony_content(tokenizer: Any, token_ids: Sequence[int], fallback_text: str) -> str:
    if not supports_harmony_postprocess(tokenizer):
        return normalize_generation_text(fallback_text)

    decoded_text = fallback_text
    if token_ids:
        try:
            decoded_text = tokenizer.decode(token_ids)
        except Exception as exc:
            logger.debug("Failed to decode Harmony token ids: %r", exc)

    if hasattr(tokenizer, "parse_response"):
        try:
            parsed = tokenizer.parse_response(decoded_text)
            content = parsed.get("content") if isinstance(parsed, dict) else None
            if isinstance(content, str) and content.strip():
                return content
        except Exception as exc:
            logger.debug("Failed to parse Harmony response text: %r", exc)

    if token_ids and hasattr(tokenizer, "parse_harmony_message"):
        try:
            response_prefill = tokenizer.encode("<|start|>assistant")
            parsed_messages = tokenizer.parse_harmony_message(response_prefill + list(token_ids))
            last_content = None
            for message in parsed_messages:
                content = getattr(message, "content", None)
                if content is not None and getattr(content, "token_ids", None):
                    text = tokenizer.decode(content.token_ids)
                    if text.strip():
                        last_content = text
            if last_content is not None:
                return last_content
        except Exception as exc:
            logger.debug("Failed to parse Harmony token stream: %r", exc)

    return normalize_generation_text(decoded_text)


def _normalize_latex_value(value: str) -> str:
    value = value.strip()
    text_match = _LATEX_TEXT_RE.match(value)
    if text_match:
        value = text_match.group(1)
    return value.strip().strip("$").strip()


def iter_boxed_values(text: str) -> Iterable[str]:
    for match in _BOXED_RE.finditer(text):
        value = _normalize_latex_value(match.group(1))
        if value:
            yield value


def parse_boxed_scalar(text: str, default: str = "") -> str:
    values = list(iter_boxed_values(extract_final_answer_segment(text)))
    return values[-1] if values else default


def iter_code_blocks(text: str, language: Optional[str] = None) -> Iterable[str]:
    text = normalize_generation_text(text)
    if language:
        language_re = re.escape(language.strip().lower())
        pattern = re.compile(rf"```{language_re}\s*\n?(.*?)\n?```", re.IGNORECASE | re.DOTALL)
    else:
        pattern = _CODE_BLOCK_RE
    for match in pattern.finditer(text):
        yield match.group(1)


def parse_code_block(text: str, language: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
    blocks = list(iter_code_blocks(text, language=language))
    return blocks[-1] if blocks else default


def _letter_class(letters: str) -> str:
    return re.escape("".join(dict.fromkeys(letters.upper())))


def parse_mcq_single(
    text: str,
    letters: str = "ABCD",
    default: str = "",
    prefer_boxed: bool = True,
    fallback_window: int = 200,
) -> str:
    final_text = extract_final_answer_segment(text).translate(_FULLWIDTH_TO_ASCII)
    letter_re = _letter_class(letters)

    if prefer_boxed:
        for value in reversed(list(iter_boxed_values(final_text))):
            match = re.fullmatch(rf"\s*[{letter_re}]\s*", value.upper())
            if match:
                return match.group(0).strip().upper()

    patterns = [
        rf"(?:the\s+)?answer\s+is\s*[\(\[\{{]?\s*([{letter_re}])\b\s*[\)\]\}}]?",
        rf"(?:exact\s+)?answer\s*:\s*(?:\\boxed)?\s*[\(\[\{{]?\s*([{letter_re}])\b\s*[\)\]\}}]?",
        rf"final\s+answer\s*:\s*[\(\[\{{]?\s*([{letter_re}])\b\s*[\)\]\}}]?",
        rf"答え(?:は|:)?\s*[\(\[\{{]?\s*([{letter_re}])\b\s*[\)\]\}}]?",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(
            (match.start(), match.group(1).upper())
            for match in re.finditer(pattern, final_text, re.IGNORECASE | re.DOTALL)
        )
    if candidates:
        return sorted(candidates, key=lambda item: item[0])[-1][1]

    tail = final_text.upper()[-fallback_window:]
    fallback_matches = re.findall(rf"(?<![A-Z])([{letter_re}])(?![A-Z])", tail)
    if fallback_matches:
        return fallback_matches[-1].upper()
    return default


def _normalize_number_token(token: str) -> Optional[str]:
    token = token.strip().lower()
    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]
    token = token.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
        return token
    return None


def parse_numeric(text: str, default: Optional[str] = None, allow_words: bool = True) -> Optional[str]:
    final_text = extract_final_answer_segment(text).translate(_FULLWIDTH_TO_ASCII)
    number_words = "|".join(_NUMBER_WORDS) if allow_words else r"(?!x)x"
    token = rf"-?\d+(?:,\d{{3}})*(?:\.\d+)?|{number_words}"

    boxed_values = list(iter_boxed_values(final_text))
    for value in reversed(boxed_values):
        normalized = _normalize_number_token(value)
        if normalized is not None:
            return normalized

    patterns = [
        rf"(?:final\s+answer|answer)\s*(?:is|:)?[^\n\r\dA-Za-z-]{{0,20}}\s*({token})\b",
        rf"\b({token})\s+sisters?\b",
    ]
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, final_text, re.IGNORECASE | re.DOTALL):
            normalized = _normalize_number_token(match.group(1))
            if normalized is not None:
                candidates.append((match.start(), normalized))
    if candidates:
        return sorted(candidates, key=lambda item: item[0])[-1][1]
    normalized = _normalize_number_token(final_text.strip())
    if normalized is not None:
        return normalized
    return default


__all__ = [
    "extract_final_answer_segment",
    "extract_harmony_content",
    "extract_reasoning_final_segment",
    "iter_boxed_values",
    "iter_code_blocks",
    "normalize_generation_text",
    "parse_boxed_scalar",
    "parse_code_block",
    "parse_mcq_single",
    "parse_numeric",
    "supports_harmony_postprocess",
    "unwrap_generation_output",
]
