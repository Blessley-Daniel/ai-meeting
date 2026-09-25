"""Optional LoRA fine-tuning of the meeting-minutes extraction model.

Read this before running it
---------------------------

Fine-tuning is **optional** and this script exists to demonstrate the method,
not because the system needs it. The running application works with the
pretrained model plus the deterministic rule layer. Before assuming
fine-tuning will help, measure: run ``evaluate.py`` on the test split with and
without the adapter and compare. If it does not improve the structured fields,
the honest conclusion is that it did not help on this dataset, and that is a
perfectly good result to report.

Why LoRA / PEFT rather than full fine-tuning
--------------------------------------------

Full fine-tuning updates every weight of the base model. That needs far more
memory than a college laptop has: for a 0.5B model, optimizer state alone is
several times the model size. LoRA (Low-Rank Adaptation) freezes the base
weights and trains small low-rank matrices beside the attention projections.
Only those are updated, so the number of trainable parameters drops by roughly
two orders of magnitude, and the adapter that comes out is a few megabytes
rather than a full copy of the model.

The base model stays untouched, which also means the adapter can be evaluated
against the exact same base weights, isolating what fine-tuning actually
changed.

Hardware requirements
---------------------

The defaults are chosen to run on a CPU-only machine:

* ``--model Qwen/Qwen2.5-0.5B-Instruct`` - about 1 GB in float32.
* ``--max-length 768`` - keeps activation memory modest.
* ``--batch-size 1`` with ``--grad-accum 8`` - effective batch of 8 without
  needing 8 examples in memory at once.
* ``--limit 200`` - a cap on steps so a run finishes in a sane time.

On a CPU-only laptop expect this to take tens of minutes for a few dozen steps.
It is slow but it does complete. If you have a CUDA GPU, pass ``--device cuda``
and it will be substantially faster. If it is still too slow, reduce
``--max-steps`` - the goal is to demonstrate the pipeline, not to reach a
target score.

Usage:
    # Quick smoke test: 4 steps, tiny model, just to prove the mechanism works
    python training/train.py --max-steps 4 --limit 8

    # A more real run
    python training/train.py --max-steps 120 --limit 200

    # Then evaluate the result
    python training/evaluate.py --split test --method hybrid
        (with extraction_adapter_path set in .env)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.extraction import SCHEMA_SPEC  # noqa: E402


def build_user_prompt(transcript: str) -> str:
    """The instruction half of a training example.

    This must match ``build_prompt`` in ``services/extraction.py``. If the two
    drift apart, the model is trained on one instruction and asked a different
    one at inference time, and the adapter will look useless for reasons that
    have nothing to do with the model.
    """
    from app.services.extraction import build_prompt

    return build_prompt(transcript)


SYSTEM_PROMPT = (
    "You extract structured minutes from meeting transcripts. "
    "Reply with a single JSON object and nothing else. "
    "Never invent information: use 'Not specified' for anything absent."
)


def format_example(example: dict, tokenizer) -> dict:
    """Turn one dataset record into a chat-formatted training example.

    Loss is computed on the assistant's reply only. Training on the prompt
    tokens as well would teach the model to reproduce transcripts, which is not
    the task and would waste most of the gradient signal.
    """
    transcript = example.get("transcript", "")
    target = example.get("target", {})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(transcript)},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]
    return {"messages": messages}


def tokenize_example(example: dict, tokenizer, max_length: int) -> dict:
    """Tokenise a chat example and mask the prompt tokens in the labels."""
    messages = format_example(example, tokenizer)["messages"]

    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    full = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=False
    )
    prompt = tokenizer(
        prompt_text, truncation=True, max_length=max_length, add_special_tokens=False
    )

    labels = list(full["input_ids"])
    prompt_length = min(len(prompt["input_ids"]), len(labels))
    # Mask the prompt so loss reflects only the JSON the model should produce.
    for index in range(prompt_length):
        labels[index] = -100

    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


class ChatDataset:
    """A minimal dataset wrapper, so no ``datasets`` dependency is needed."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int):
        self.rows = [tokenize_example(e, tokenizer, max_length) for e in examples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class Collator:
    """Pad a batch to the length of its longest sequence."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            # Pad labels with -100 so padding contributes no loss.
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for extraction.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "models" / "mom-lora",
    )
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--limit", type=int, default=200, help="Cap training examples.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=20)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)

    train_path = REPO_ROOT / "datasets" / "processed" / "train.jsonl"
    val_path = REPO_ROOT / "datasets" / "processed" / "validation.jsonl"
    if not train_path.exists():
        raise SystemExit(
            f"{train_path} not found. Run: python training/prepare_dataset.py"
        )

    train_examples = [
        json.loads(line) for line in train_path.read_text().splitlines() if line.strip()
    ][: args.limit]
    val_examples = (
        [json.loads(line) for line in val_path.read_text().splitlines() if line.strip()]
        if val_path.exists()
        else []
    )

    print(f"Base model           : {args.model}")
    print(f"Device               : {args.device}")
    print(f"Train / validation   : {len(train_examples)} / {len(val_examples)}")
    print(f"Steps                : {args.max_steps}")
    print(f"Effective batch size : {args.batch_size * args.grad_accum}\n")

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)

    # LoRA is applied to the projection layers of attention, which is where
    # task-specific behaviour is cheapest to steer.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    dataset = ChatDataset(train_examples, tokenizer, args.max_length)
    val_dataset = (
        ChatDataset(val_examples, tokenizer, args.max_length) if val_examples else None
    )
    collator = Collator(tokenizer)

    if len(dataset) == 0:
        raise SystemExit("No training examples available.")

    model.train()
    step = 0
    epoch = 0
    running_loss = 0.0
    micro_step = 0

    while step < args.max_steps:
        epoch += 1
        print(f"--- epoch {epoch} ---")
        for start in range(0, len(dataset), args.batch_size):
            if step >= args.max_steps:
                break
            batch = collator(
                [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
            )
            batch = {key: value.to(args.device) for key, value in batch.items()}

            outputs = model(**batch)
            # Normalise by accumulation steps so the gradient scale does not
            # depend on --grad-accum.
            loss = outputs.loss / args.grad_accum
            loss.backward()

            running_loss += float(outputs.loss.detach())
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                average = running_loss / (micro_step or 1)
                print(f"  step {step:4d}/{args.max_steps}  loss {average:.4f}")
                running_loss = 0.0
                micro_step = 0

                if step % args.eval_every == 0 and val_dataset is not None:
                    model.eval()
                    with torch.no_grad():
                        val_batch = collator([val_dataset[0]])
                        val_batch = {
                            key: value.to(args.device) for key, value in val_batch.items()
                        }
                        val_loss = float(model(**val_batch).loss)
                    print(f"           validation loss {val_loss:.4f}")
                    model.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Record what produced this adapter, so a result can be attributed later.
    (args.output_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "method": "LoRA/PEFT",
                "max_steps": args.max_steps,
                "train_examples": len(train_examples),
                "validation_examples": len(val_examples),
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "effective_batch_size": args.batch_size * args.grad_accum,
                "device": args.device,
                "dataset": "synthetic (generated by training/prepare_dataset.py)",
                "schema": SCHEMA_SPEC,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nAdapter saved to {args.output_dir}")
    print(
        "To use it, set extraction_adapter_path in .env to that directory, "
        "then compare evaluation scores with and without it."
    )


if __name__ == "__main__":
    main()