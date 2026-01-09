import os
import torch
import argparse
from evaluation.domain_zero_shot import evaluate_hqs, evaluate_mednli, evaluate_pubmedqa
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():

    parser = argparse.ArgumentParser(description="Evaluating model on medical datasets.")
    parser.add_argument("--model", type=str, default="model/downloaded/there_is_no_name", help="Path to the fine-tuned model.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility.")
    parser.add_argument("--log_dir", type=str, help="Path to save the evaluation log file.")
    parser.add_argument("--filename", type=str, default="med_eval_results.txt", help="Filename for the evaluation results.")
    

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    log_path = os.path.join(args.log_dir, args.filename)
    os.makedirs(args.log_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.model)

    with open(log_path, "w") as log_file:

        acc, maf, cm, predictions = evaluate_pubmedqa(args.model,n_eval=500, seed=args.seed)
        log_file.write("PubMedQA Results with seed {}:\n".format(args.seed))
        log_file.write("Accuracy: {:.4f}, Macro F1: {:.4f}\n".format(acc, maf))
        log_file.write("\n")

        acc, maf, cm, predictions = evaluate_mednli(args.model, seed=args.seed)
        log_file.write("MedNLI Results with seed {}:\n".format(args.seed))
        log_file.write("Accuracy: {:.4f}, Macro F1: {:.4f}\n".format(acc, maf))
        log_file.write("\n")

        rouge_scores, summaries = evaluate_hqs(args.model, n_eval=100, seed=args.seed)
        log_file.write("HQS Results with seed {}:\n".format(args.seed))
        log_file.write("ROUGE-1 Score: {:.4f}\n".format(rouge_scores['rouge1'])) 
        log_file.write("ROUGE-2 Score: {:.4f}\n".format(rouge_scores['rouge2']))
        log_file.write("ROUGE-L Score: {:.4f}\n".format(rouge_scores['rougeL']))
        log_file.write("\n")
