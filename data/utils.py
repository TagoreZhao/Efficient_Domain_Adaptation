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
        "prompt": [{"role": "user", "content": "..."}],
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
          "prompt": [{"role": "user", "content": "..."}],
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

def to_eval_loss_dataset(data_list, response_template="Response: The answer is"):
    """
    Converts a list of dicts with keys:
      - 'input_text'  (prompt that ends with the response prefix)
      - 'target_text' (gold completion, e.g., ' yes')
    into a Dataset with a single 'text' field:
      text = input_text + ' ' + normalized target_text

    Use together with TRL's DataCollatorForCompletionOnlyLM configured
    with the same response_template so that labels are created only
    for the target span.
    """
    rows = []
    for ex in data_list:
        prompt = str(ex["input_text"]).rstrip()              # ensure single trailing space
        target = str(ex["target_text"]).lstrip()             # strip leading spaces like ' yes'
        rows.append({"text": f"{prompt} {target}"})
    return Dataset.from_list(rows)