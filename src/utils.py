import re
from typing import Dict, List, Union

from prompts import (  # noqa: F401
    PLAIN_SYSTEM_PROMPT, COT_SYSTEM_PROMPT, NTCOT_SYSTEM_PROMPT,
    CLAUDE_REINFORCEMENT_PROMPT,
    SUMMARIZE, FEW_SHOT_EXAMPLES, FEW_SHOT_EXAMPLES_COT, OPUS_PLAIN_FEW_SHOT_EXAMPLES,
)

# NOTE: Do not use underscores in short names. This can break experiment parsing.
BEDROCK_MODELS = {
    "us.anthropic.claude-opus-4-5-20251101-v1:0": "claude-opus-45",
    "us.anthropic.claude-sonnet-4-20250514-v1:0": "claude-sonnet-4",
    "us.meta.llama3-3-70b-instruct-v1:0": "llama3.3-70b",
    "deepseek.v3-v1:0": "deepseek-v31",
    "openai.gpt-oss-120b-1:0": "gpt-oss-120b",
    "openai.gpt-oss-20b-1:0": "gpt-oss-20b",
    "us.meta.llama4-maverick-17b-instruct-v1:0": "llama4-maverick-17b",
    "us.meta.llama4-scout-17b-instruct-v1:0": "llama4-scout-17b",
}

MODELS_SHORT = dict(BEDROCK_MODELS)
MODELS_LONG = {v: k for k, v in MODELS_SHORT.items()}


def generate_short_folder_name(model: str, use_cot: bool = None, ntcot: bool = None) -> str:
    """
    Generate a short, human-readable folder name based on model and experiment parameters.

    Args:
        model (str): The Bedrock model ID (e.g., "us.meta.llama3-3-70b-instruct-v1:0")
        use_cot (bool, optional): Whether chain-of-thought mode is used
        ntcot (bool, optional): Whether non-therapist COT mode is used

    Returns:
        str: Short folder name (e.g., "llama3.3-70b_COT", "claude-opus-45_NTCOT")
    """
    short_name = MODELS_SHORT.get(model, model)

    suffix = ""
    if ntcot is True:
        suffix += "_NTCOT"
    elif use_cot is True:
        suffix += "_COT"

    return short_name + suffix


# Matches <tag ...>...</tag> blocks (any tag name; preserves inner content exactly)
_TAG_BLOCK = re.compile(
    r"<\s*([A-Za-z][A-Za-z0-9_\-]*)\b[^>]*>(.*?)</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

def extract_blocks_to_dict(text: str) -> Dict[str, str]:
    """
    Extract raw contents of <tag>...</tag> blocks with NO transformation of inner text.
    - Renames the 'final' tag to 'response'.
    - If 'response' is present and its content is empty (after whitespace strip),
      set 'response' to the entire original input text.
    - Single occurrence -> string; multiple occurrences -> list of strings.
    Returns a dictionary.
    """
    if not isinstance(text, str) or not text:
        return {"response": ""}

    buckets: Dict[str, List[str]] = {}
    for m in _TAG_BLOCK.finditer(text):
        tag = m.group(1).lower()
        key = "response" if tag == "final" else tag
        content = m.group(2)  # preserved exactly
        buckets.setdefault(key, []).append(content)

    # Collapse singletons
    result: Dict[str, Union[str, List[str]]] = {
        k: v[0] if len(v) == 1 else v for k, v in buckets.items()
    }

    # If response does not exist or is empty, replace with the entire original input text
    if "response" in result:
        val = result["response"]
        if isinstance(val, list):
            # Take the last occurrence if there are multiple
            result["response"] = val[-1] if val else ""
        elif isinstance(val, str):
            if val.strip() == "":
                result["response"] = text
        else:
            result["response"] = str(val)

    else:
        result["response"] = text

    return result


def prompt_to_messages(prompt, use_cot: bool = False,
                       cot_prompt: str = COT_SYSTEM_PROMPT, reinforcement_prompt: str = None,
                       opus_plain_mode: bool = False):
    if use_cot:
        system = cot_prompt
        few_shot = FEW_SHOT_EXAMPLES_COT
    elif opus_plain_mode:
        system = PLAIN_SYSTEM_PROMPT
        few_shot = OPUS_PLAIN_FEW_SHOT_EXAMPLES
    else:
        system = PLAIN_SYSTEM_PROMPT
        few_shot = FEW_SHOT_EXAMPLES

    if reinforcement_prompt:
        system += "\n" + reinforcement_prompt
    messages = [{"role": "system", "content": system}]
    messages += few_shot
    messages += [{"role": "user", "content": prompt}]
    # Assistant prefill for Opus 4.5 plain mode: forces the model to continue from inside the tag
    if opus_plain_mode:
        messages.append({"role": "assistant", "content": "<final>"})
    return messages
