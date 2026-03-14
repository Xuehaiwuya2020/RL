"""
Minimal SFT training demo for multi-round dialogue data.

This script is intentionally simple and only keeps the minimum pieces to run:
- load tokenizer/model
- build dataset/dataloader
- run a few training steps
- save model/tokenizer
"""

import argparse
import logging
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_set import MultiRoundDialogueDataset, collate_func


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal SFT demo")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--data_dir", type=str, default="./cache")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite_cache", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def train_one_epoch(model, dataloader, optimizer, device, max_steps):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader, start=1):
        if step > max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 5 == 0 or step == 1:
            logger.info("step=%d loss=%.6f", step, loss.item())

    return total_loss / max(1, min(len(dataloader), max_steps))


def main():
    args = parse_args()

    device = resolve_device(args.device)
    logger.info("device=%s", device)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model.to(device)

    train_dataset = MultiRoundDialogueDataset(
        tokenizer=tokenizer,
        max_len=args.max_len,
        data_dir=args.data_dir,
        data_set_name="train_demo",
        path_file=args.train_file,
        is_overwrite=args.overwrite_cache,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_func,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    logger.info("dataset_size=%d", len(train_dataset))
    logger.info("steps_per_epoch=%d (capped by --max_steps=%d)", len(train_dataloader), args.max_steps)

    for epoch in range(1, args.num_epochs + 1):
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, device, args.max_steps)
        logger.info("epoch=%d avg_loss=%.6f", epoch, avg_loss)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
