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
    parser.add_argument("--output_path", type=str, default="data/downloaded/med_dataset", help="Path to save the constructed dataset.")
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

    if "qwen3" in str(args.model).lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        qwen3_dataset = to_prompt_completion_hf(dataset, tokenizer=tokenizer)
        qwen3_dataset.save_to_disk(args.output_path)
    elif "llama2" in str(args.model).lower():
        print("Llama2 dataset construction not implemented yet.")

        