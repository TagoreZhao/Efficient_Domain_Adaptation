"""
Smoke test for prune_sparsegpt with meta-llama/Llama-3.2-1B.

Usage:
    export PYTHONPATH=.:$PYTHONPATH
    python test_sparsegpt.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pruning.prune import prune_sparsegpt
from pruning.utils import check_sparsity, get_layers, find_layers
from evaluation.perplexity import eval_ppl

MODEL_NAME = "meta-llama/Llama-3.2-1B"
SPARSITY_RATIO = 0.5
NSAMPLES = 128
SEQLEN = 2048
SEED = 0
DATASET_NAME = "c4"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model and tokenizer ---
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    ).to(device)

    seqlen = SEQLEN
    print(f"Using seqlen={seqlen}")

    # --- Verify model-agnostic utilities ---
    layers = get_layers(model)
    print(f"get_layers() found {len(layers)} decoder layers.")
    subset = find_layers(layers[0])
    print(f"find_layers() found {len(subset)} linear layers in layer 0: {list(subset.keys())}")

    # --- Sparsity before pruning ---
    sparsity_before = check_sparsity(model)
    print(f"\nSparsity before pruning: {sparsity_before:.6f}")
    assert sparsity_before < 0.01, f"Expected near-zero sparsity before pruning, got {sparsity_before}"

    # --- Run SparseGPT pruning ---
    print(f"\nRunning prune_sparsegpt (sparsity={SPARSITY_RATIO}, nsamples={NSAMPLES}, seqlen={seqlen})...")
    result = prune_sparsegpt(
        sparsity_ratio=SPARSITY_RATIO,
        nsamples=NSAMPLES,
        seed=SEED,
        seqlen=seqlen,
        model=model,
        tokenizer=tokenizer,
        dataset_name=DATASET_NAME,
        device=device,
    )
    assert result is model, "prune_sparsegpt should return the same model object"

    # --- Sparsity after pruning ---
    sparsity_after = check_sparsity(model)
    print(f"\nSparsity after pruning: {sparsity_after:.6f}")
    print(f"Target sparsity:       {SPARSITY_RATIO:.6f}")

    tolerance = 0.05
    assert abs(sparsity_after - SPARSITY_RATIO) < tolerance, (
        f"Sparsity {sparsity_after:.4f} not within {tolerance} of target {SPARSITY_RATIO}"
    )

    # --- WikiText2 perplexity ---
    print(f"\nEvaluating WikiText2 perplexity (seqlen={seqlen})...")
    ppl = eval_ppl("wikitext2", seqlen, model, tokenizer, device=device, seed=SEED)
    print(f"WikiText2 perplexity: {ppl:.2f}")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
