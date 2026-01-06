from datasets import Dataset, Features, Value, Sequence
from typing import Any, Dict, List, Optional
from trl import apply_chat_template

def to_prompt_completion_conversation(
    dataset: List[Dict[str, Any]],
    input_key: str = "input_text",
    target_key: str = "target_text",
) -> List[Dict[str, Any]]:
    """
    Convert a list of dicts with keys like input_text/target_text into:
      {
        "prompt": [{"role": "user", "content": "..."}] (optionally preceded by system),
        "completion": [{"role": "assistant", "content": "..."}]
      }

    If data[input_key] or data[target_key] are already in list-of-dicts chat format,
    they will be used as-is (with minimal validation).
    """
    normalized: List[Dict[str, Any]] = []

    for i, data in enumerate(dataset):

        normalized.append({
            "prompt": [{"role": str("user"), "content": str(data[input_key])}],
            "completion": [{"role": str("assistant"), "content": str(data[target_key])}],
        })
 
    return normalized

def to_prompt_completion_hf(
    raw_data: List[Dict[str, Any]],
    tokenizer: Any,
    input_key: str = "input_text",
    target_key: str = "target_text",
    drop_invalid: bool = True,
) -> Dataset:
    """
    Convert a list/sequence of raw examples into a Hugging Face Dataset in
    prompt/completion chat-list format:

        {
          "prompt": [{"role": "user", "content": "..."}] (optionally preceded by system),
          "completion": [{"role": "assistant", "content": "..."}],
        }

    Args:
        raw_data: Raw records (list/sequence of dicts).
        tokenizer: Tokenizer to use for applying chat template.
        input_key: Key in each record containing the user prompt text/messages.
        target_key: Key in each record containing the assistant completion text/messages.
        system_prompt: Optional system message to prepend to each prompt.
        drop_invalid: If True, silently drop malformed examples; otherwise raise.

    Returns:
        A `datasets.Dataset` with the provided `features`.
    """
    formatted = to_prompt_completion_conversation(
        raw_data,
        input_key=input_key,
        target_key=target_key,
    )

    if not drop_invalid:
        # Minimal structural validation against expected columns
        for i, ex in enumerate(formatted):
            if "prompt" not in ex or "completion" not in ex:
                raise ValueError(f"Formatted example {i} missing 'prompt'/'completion' keys: {ex!r}")

    dataset = Dataset.from_list(formatted)
    return dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})