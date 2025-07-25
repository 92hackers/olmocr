"""
Simple script to test OlmOCR dataset loading with YAML configuration.
"""

import argparse
import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from olmocr.train.config import Config
from olmocr.train.dataloader import BaseMarkdownPDFDataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class QwenDataCollator:
    """Data collator for vision-language models that handles numpy arrays."""

    def __init__(self, max_token_len: Optional[int] = None):
        self.max_token_len = max_token_len

    def __call__(self, examples):
        # Filter out None values and extract the fields we need
        batch = {"input_ids": [], "attention_mask": [], "labels": [], "pixel_values": [], "image_grid_thw": []}

        for example in examples:
            if example is not None:
                # Convert numpy arrays to tensors
                input_ids = torch.from_numpy(example["input_ids"]) if isinstance(example["input_ids"], np.ndarray) else example["input_ids"]
                attention_mask = torch.from_numpy(example["attention_mask"]) if isinstance(example["attention_mask"], np.ndarray) else example["attention_mask"]
                labels = torch.from_numpy(example["labels"]) if isinstance(example["labels"], np.ndarray) else example["labels"]

                # Trim to max_token_len if specified
                if self.max_token_len is not None:
                    input_ids = input_ids[: self.max_token_len]
                    attention_mask = attention_mask[: self.max_token_len]
                    labels = labels[: self.max_token_len]

                batch["input_ids"].append(input_ids)
                batch["attention_mask"].append(attention_mask)
                batch["labels"].append(labels)

                # Handle pixel_values which might be numpy array or already a tensor
                pixel_values = example["pixel_values"]
                if isinstance(pixel_values, np.ndarray):
                    pixel_values = torch.from_numpy(pixel_values)
                batch["pixel_values"].append(pixel_values)

                # Handle image_grid_thw
                image_grid_thw = example["image_grid_thw"]
                if isinstance(image_grid_thw, np.ndarray):
                    image_grid_thw = torch.from_numpy(image_grid_thw)
                batch["image_grid_thw"].append(image_grid_thw)

        # Convert lists to tensors with proper padding
        # Note: For Qwen2-VL, we typically handle variable length sequences
        # The model's processor should handle the padding internally
        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": torch.stack(batch["labels"]),
            "pixel_values": torch.stack(batch["pixel_values"]),  # Stack into tensor
            "image_grid_thw": torch.stack(batch["image_grid_thw"]),
        }


def main():
    parser = argparse.ArgumentParser(description="Train OlmOCR model")
    parser.add_argument("--config", type=str, default="olmocr/train/configs/example_config.yaml", help="Path to YAML configuration file")

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return

    # Set wandb project from config
    if config.project_name:
        os.environ["WANDB_PROJECT"] = config.project_name
        logger.info(f"Setting WANDB_PROJECT to: {config.project_name}")

    # Load processor for tokenization
    logger.info(f"Loading processor: {config.model.name}")
    processor = AutoProcessor.from_pretrained(
        config.model.name,
    )

    # Load model
    logger.info(f"Loading model: {config.model.name}")
    if "Qwen2.5-VL" in config.model.name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model.name,
            torch_dtype=getattr(torch, config.model.torch_dtype) if config.model.torch_dtype != "auto" else "auto",
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
            attn_implementation=config.model.attn_implementation if config.model.use_flash_attention else None,
        )
    elif "Qwen2-VL" in config.model.name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model.name,
            torch_dtype=getattr(torch, config.model.torch_dtype) if config.model.torch_dtype != "auto" else "auto",
            device_map=config.model.device_map,
            trust_remote_code=config.model.trust_remote_code,
            attn_implementation=config.model.attn_implementation if config.model.use_flash_attention else None,
        )
    else:
        raise NotImplementedError()

    # Enable gradient checkpointing if configured
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=config.training.gradient_checkpointing_kwargs)

    # Create training datasets
    logger.info("Creating training datasets...")
    train_datasets = []
    for i, dataset_cfg in enumerate(config.dataset.train):
        root_dir = dataset_cfg["root_dir"]
        pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)

        logger.info(f"Creating training dataset {i+1} from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            train_datasets.append(dataset)

    # Combine all training datasets
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    logger.info(f"Total training samples: {len(train_dataset)}")

    # Create evaluation datasets
    logger.info("Creating evaluation datasets...")
    eval_datasets = {}
    for i, dataset_cfg in enumerate(config.dataset.eval):
        root_dir = dataset_cfg["root_dir"]
        pipeline_steps = config.get_pipeline_steps(dataset_cfg["pipeline"], processor)

        # Use dataset name if provided, otherwise use root_dir as name
        dataset_name = dataset_cfg.get("name", f"eval_dataset_{i+1}")

        logger.info(f"Creating evaluation dataset '{dataset_name}' from: {root_dir}")
        dataset = BaseMarkdownPDFDataset(root_dir, pipeline_steps)
        logger.info(f"Found {len(dataset)} samples")

        if len(dataset) > 0:
            eval_datasets[dataset_name] = dataset

    # Log total evaluation samples across all datasets
    total_eval_samples = sum(len(dataset) for dataset in eval_datasets.values())
    logger.info(f"Total evaluation samples across {len(eval_datasets)} datasets: {total_eval_samples}")

    # Construct full output directory by appending run_name to base output_dir
    full_output_dir = os.path.join(config.training.output_dir, config.run_name)
    logger.info(f"Setting output directory to: {full_output_dir}")

    # Check for existing checkpoints if any
    found_resumable_checkpoint = False
    if os.path.exists(full_output_dir):
        # Look for checkpoint directories
        checkpoint_dirs = [d for d in os.listdir(full_output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(full_output_dir, d))]
        if checkpoint_dirs:
            # Sort by checkpoint number and get the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
            latest_checkpoint = os.path.join(full_output_dir, checkpoint_dirs[-1])
            logger.info(f"Found existing checkpoint: {latest_checkpoint}")
            found_resumable_checkpoint = latest_checkpoint
        else:
            logger.info("No existing checkpoints found in output directory")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=full_output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=float(config.training.learning_rate),
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_kwargs=config.training.lr_scheduler_kwargs,
        optim=config.training.optim,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        adam_epsilon=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        bf16=True,  # We're sticking with this known good reduced precision option
        eval_strategy=config.training.evaluation_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        logging_dir=config.training.logging_dir,
        logging_strategy=config.training.logging_strategy,
        logging_steps=config.training.logging_steps,
        logging_first_step=config.training.logging_first_step,
        report_to=config.training.report_to,
        seed=config.training.seed,
        data_seed=config.training.data_seed,
        push_to_hub=False,
        label_names=["labels"],
        dataloader_drop_last=config.training.dataloader_drop_last,
        dataloader_num_workers=config.training.dataloader_num_workers,
        remove_unused_columns=config.training.remove_unused_columns,
        eval_on_start=True,
        run_name=config.run_name,
    )

    # Set up callbacks
    callbacks = []
    if config.training.use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience, early_stopping_threshold=config.training.early_stopping_threshold
            )
        )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=QwenDataCollator(max_token_len=config.training.collator_max_token_len),
        callbacks=callbacks,
    )

    # Start training
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=found_resumable_checkpoint)

    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()

    # Log metrics
    logger.info(f"Training completed! Metrics: {train_result.metrics}")


if __name__ == "__main__":
    main()
