import os
import torch
import argparse
from evaluation.perplexity import eval_ppl
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.domain_zero_shot import evaluate_billsum, evaluate_contractnli, evaluate_casehold


def parse_args():
    
    parser = argparse.ArgumentParser(description="Evaluating model on legal datasets.")
    parser.add_argument("--model", type=str, default="model/downloaded/there_is_no_name", help="Path to the fine-tuned model.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility.")
    parser.add_argument("--log_dir", type=str, help="Path to save the evaluation log file.")
    parser.add_argument("--filename", type=str, default="legal_eval_results.txt", help="Filename for the evaluation results.")
    

    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    
    log_path = os.path.join(args.log_dir, args.filename)
    os.makedirs(args.log_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if torch.cuda.is_available():
        model.to(torch.device("cuda:0"))
    device = model.device
    with open(log_path, "w") as log_file:

        ppl = eval_ppl(model=model, tokenizer=tokenizer, dataset='legal_case_reports', seqlen=2048, device=device, seed=args.seed)
        print(f"Perplexity on Legal Case Reports with seed {args.seed}: {ppl:.4f}")
        log_file.write("Perplexity on Legal Case Reports with seed {}: {:.4f}\n".format(args.seed, ppl))
        log_file.write("\n")

        acc, maf, cm, predictions = evaluate_contractnli(model, seed=args.seed, tokenizer=tokenizer, device=device)
        print(f"ContractNLI Results with seed {args.seed}: Accuracy: {acc:.4f}, Macro F1: {maf:.4f}")
        log_file.write("ContractNLI Results with seed {}:\n".format(args.seed))
        log_file.write("Accuracy: {:.4f}, Macro F1: {:.4f}\n".format(acc, maf))
        log_file.write("\n")

        acc, maf, cm, predictions = evaluate_casehold(model, seed=args.seed, tokenizer=tokenizer, device=device)
        print(f"CaseHold Results with seed {args.seed}: Accuracy: {acc:.4f}, Macro F1: {maf:.4f}")
        log_file.write("CaseHold Results with seed {}:\n".format(args.seed))
        log_file.write("Accuracy: {:.4f}, Macro F1: {:.4f}\n".format(acc, maf))
        log_file.write("\n")

        rouge_scores, summaries = evaluate_billsum(model, n_eval=100, seed=args.seed, tokenizer=tokenizer, device=device)
        print(f"BillSum Results with seed {args.seed}: ROUGE-1: {rouge_scores['rouge1']:.4f}, ROUGE-2: {rouge_scores['rouge2']:.4f}, ROUGE-L: {rouge_scores['rougeL']:.4f}")
        log_file.write("BillSum Results with seed {}:\n".format(args.seed))
        log_file.write("ROUGE-1 Score: {:.4f}\n".format(rouge_scores['rouge1'])) 
        log_file.write("ROUGE-2 Score: {:.4f}\n".format(rouge_scores['rouge2']))
        log_file.write("ROUGE-L Score: {:.4f}\n".format(rouge_scores['rougeL']))
        log_file.write("\n")    