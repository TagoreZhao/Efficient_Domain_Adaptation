import argparse
from data.datasets import construct_legal_data
from data.utils import to_prompt_completion_hf
from transformers import AutoTokenizer

def parse_args():

    parser = argparse.ArgumentParser(description="Construct legal datasets and save in HuggingFace format.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to construct (train/val).")
    parser.add_argument("--casehold_count", type=int, default=7000, help="Number of Case Law examples for training.")
    parser.add_argument("--casehold_prop", type=list, default=[1/5, 1/5, 1/5, 1/5, 1/5], help="Proportion of Case Law examples for train/val/test splits.")
    parser.add_argument("--billsum_count", type=int, default=2000, help="Number of Statutes examples for training.")
    parser.add_argument("--contractnli_count", type=int, default=6000, help="Number of Contracts examples for training.")
    parser.add_argument("--entailment_prop", type=float, default=0.35, help="Ratio of 'entailment' examples in the dataset.")
    parser.add_argument("--contradiction_prop", type=float, default=0.2, help="Ratio of 'contradiction' examples in the dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, default="data/downloaded/qwen_legal_dataset", help="Path to save the constructed dataset.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model type for which the dataset is being constructed.")

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()

    dataset = construct_legal_data(
        split=args.split,
        casehold_count=args.casehold_count,
        casehold_prop=args.casehold_prop,
        billsum_count=args.billsum_count,
        contractnli_count=args.contractnli_count,
        entailment_prop=args.entailment_prop,
        c4_count= 0,
        contradiction_prop=args.contradiction_prop,
        seed=args.seed
    )

    if "qwen3" in str(args.model).lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        qwen3_dataset = to_prompt_completion_hf(dataset, tokenizer=tokenizer)
        qwen3_dataset.save_to_disk(args.output_path)

    elif "llama2" in str(args.model).lower():
        print("Llama2 dataset construction not implemented yet.")