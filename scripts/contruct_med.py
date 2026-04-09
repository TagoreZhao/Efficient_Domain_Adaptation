import argparse
from data.datasets import construct_med_data
from data.utils import to_prompt_completion_hf
from transformers import AutoTokenizer
def parse_args():

    parser = argparse.ArgumentParser(description="Construct medical datasets and save in HuggingFace format.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to construct (train/val).")
    parser.add_argument("--pubmed_count", type=int, default=7000, help="Number of PubMed examples for training.")
    parser.add_argument("--pubmed_yes", type=float, default=0.5, help="Ratio of 'yes' answers in the dataset.")
    parser.add_argument("--pubmed_no", type=float, default=0.5, help="Ratio of 'no' answers in the dataset.")
    parser.add_argument("--mednli_count", type=int, default=7000, help="Number of MedNLI examples for training.")
    parser.add_argument("--mednli_entailment", type=float, default=0.33, help="Ratio of 'entailment' examples in the dataset.")
    parser.add_argument("--mednli_contradiction", type=float, default=0.33, help="Ratio of 'contradiction' examples in the dataset.")
    parser.add_argument("--hqs_count", type=int, default=1000, help="Number of HQS examples for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default="data/downloaded/qwen_med_dataset", help="Path to save the constructed dataset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model type for which the dataset is being constructed.")

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()

    dataset = construct_med_data(
        split=args.split,
        pubmed_count=args.pubmed_count,
        pubmed_yes=args.pubmed_yes,
        pubmed_no=args.pubmed_no,
        mednli_count=args.mednli_count,
        mednli_entailment=args.mednli_entailment,
        mednli_contradiction=args.mednli_contradiction,
        hqs_count=args.hqs_count,
        seed=args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.chat_template is None:
        # Base Llama models lack a chat template -- borrow from the Instruct variant
        model_name = str(args.model)
        if "llama-2" in model_name.lower():
            instruct_name = model_name.replace("-hf", "-chat-hf")
        else:
            instruct_name = model_name + "-Instruct"
        print(f"No chat template found for '{args.model}'. "
              f"Borrowing from '{instruct_name}'.")
        instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_name)
        tokenizer.chat_template = instruct_tokenizer.chat_template
        print(f"Chat template borrowed successfully from '{instruct_name}'.")

    hf_dataset = to_prompt_completion_hf(dataset, tokenizer=tokenizer)
    hf_dataset.save_to_disk(args.output_path)

