from datasets import Dataset, Features, Value, Sequence
from typing import Any, Dict, List, Optional


def to_prompt_completion(
    dataset: List[Dict[str, Any]],
    input_key: str = "input_text",
    target_key: str = "target_text",
    system_prompt: Optional[str] = None,
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
        if input_key not in data or target_key not in data:
            continue

        prompt_raw = data[input_key]
        completion_raw = data[target_key]

        # Build prompt
        if isinstance(prompt_raw, list):
            # Assume already like [{"role": "...", "content": "..."}]
            prompt_msgs = prompt_raw
        else:
            prompt_msgs = []
            if system_prompt:
                prompt_msgs.append({"role": "system", "content": str(system_prompt)})
            prompt_msgs.append({"role": "user", "content": str(prompt_raw)})

        # Build completion
        if isinstance(completion_raw, list):
            completion_msgs = completion_raw
        else:
            completion_msgs = [{"role": "assistant", "content": str(completion_raw)}]

        # Minimal validation: must have required keys
        def valid_msgs(msgs: Any) -> bool:
            return (
                isinstance(msgs, list)
                and all(isinstance(m, dict) and "role" in m and "content" in m for m in msgs)
                and len(msgs) > 0
            )

        if not valid_msgs(prompt_msgs) or not valid_msgs(completion_msgs):
            continue

        normalized.append({
            "prompt": [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt_msgs],
            "completion": [{"role": str(m["role"]), "content": str(m["content"])} for m in completion_msgs],
        })
 
    return normalized

def to_prompt_completion_hf(
    dataset: List[Dict[str, Any]],
    features: Features,
    input_key: str = "input_text",
    target_key: str = "target_text",
    system_prompt: Optional[str] = None,
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
        dataset: Raw records (list/sequence of dicts).
        features: Hugging Face Features schema for the output dataset.
        input_key: Key in each record containing the user prompt text/messages.
        target_key: Key in each record containing the assistant completion text/messages.
        system_prompt: Optional system message to prepend to each prompt.
        drop_invalid: If True, silently drop malformed examples; otherwise raise.

    Returns:
        A `datasets.Dataset` with the provided `features`.
    """
    formatted = to_prompt_completion(
        list(dataset),
        input_key=input_key,
        target_key=target_key,
        system_prompt=system_prompt,
    )

    if not drop_invalid:
        # Minimal structural validation against expected columns
        for i, ex in enumerate(formatted):
            if "prompt" not in ex or "completion" not in ex:
                raise ValueError(f"Formatted example {i} missing 'prompt'/'completion' keys: {ex!r}")

    return Dataset.from_list(formatted, features=features)