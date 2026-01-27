import argparse
import hashlib
import json
import math
import os
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from specforge import (
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
    QwenVLOnlineEagle3Model,
)
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.data import (
    build_eagle3_dataset,
    build_offline_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_group,
    init_distributed,
)
from specforge.modeling.target import (
    Eagle3TargetModel,
    TargetHead,
    get_eagle3_target_model,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import Tracker, create_tracker, get_tracker_class
from specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)


# === External Schedule Support ===
# Default exponential decay base for position-based loss weighting
DEFAULT_LOSS_DECAY_BASE = 0.8


@dataclass
class ExternalSchedule:
    """Manager for external training schedules (loss position weights).

    Handles loading and querying of position-based loss weighting schedules.
    Schedule format (YAML):
        loss_position_weights:
          type: "step_based"
          schedule:
            - step_range: [0, 5000]
              weights: [1.0, 0.8, 0.64, 0.51, 0.41, 0.33, 0.26]
            - step_range: [5000, null]
              weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    """
    schedule_path: Optional[str] = None
    loss_entries: List[Dict[str, Any]] = field(default_factory=list)
    checksum: str = ""
    description: str = ""
    default_base: float = DEFAULT_LOSS_DECAY_BASE  # Configurable decay base

    @classmethod
    def from_file(cls, schedule_path: str, default_base: float = DEFAULT_LOSS_DECAY_BASE) -> "ExternalSchedule":
        """Load schedule from a YAML file."""
        import yaml

        if not os.path.exists(schedule_path):
            raise FileNotFoundError(f"Schedule file not found: {schedule_path}")

        with open(schedule_path) as f:
            config = yaml.safe_load(f)

        # Compute checksum
        config_json = json.dumps(config, sort_keys=True)
        checksum = hashlib.sha256(config_json.encode()).hexdigest()[:16]

        # Parse loss position weights
        loss_entries = []
        if "loss_position_weights" in config:
            loss_config = config["loss_position_weights"]
            if loss_config.get("type") == "step_based":
                loss_entries = loss_config.get("schedule", [])

        return cls(
            schedule_path=schedule_path,
            loss_entries=loss_entries,
            checksum=checksum,
            description=config.get("description", ""),
            default_base=default_base,
        )

    @classmethod
    def default(cls, default_base: float = DEFAULT_LOSS_DECAY_BASE) -> "ExternalSchedule":
        """Create default schedule (exponential decay base^i)."""
        return cls(
            schedule_path=None,
            loss_entries=[],
            checksum="default",
            description=f"Default exponential decay ({default_base}^i)",
            default_base=default_base,
        )

    def get_loss_weights(self, step: int, num_positions: int) -> List[float]:
        """Get position-based loss weights for a given training step."""
        if not self.loss_entries:
            return [self.default_base ** i for i in range(num_positions)]

        # Find applicable entry
        for entry in self.loss_entries:
            step_range = entry.get("step_range", [0, None])
            step_start = step_range[0] if step_range[0] is not None else 0
            step_end = step_range[1]

            if step >= step_start and (step_end is None or step < step_end):
                weights = entry.get("weights", [1.0])
                # Pad or truncate to num_positions
                if len(weights) < num_positions:
                    weights = weights + [weights[-1]] * (num_positions - len(weights))
                elif len(weights) > num_positions:
                    weights = weights[:num_positions]
                return weights

        # Fallback to last entry or default
        if self.loss_entries:
            weights = self.loss_entries[-1].get("weights", [1.0])
            if len(weights) < num_positions:
                weights = weights + [weights[-1]] * (num_positions - len(weights))
            elif len(weights) > num_positions:
                weights = weights[:num_positions]
            return weights

        return [1.0] * num_positions

    def to_checkpoint_state(self) -> Dict[str, Any]:
        """Get schedule state for checkpoint."""
        return {
            "schedule_path": self.schedule_path,
            "schedule_checksum": self.checksum,
        }


def parse_args() -> Tuple[ArgumentParser, Namespace]:
    """
    This function is used to parse the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--draft-model-config",
        type=str,
        required=False,
        help="Draft model config path. If not provided, will auto-generate from target model.",
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default="lm_head.weight",
        help="The key of the lm head weight to load from the target model, this is only required for offline training",
    )
    model_group.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM"
    )
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
        choices=["sglang", "hf", "custom"],
        help="The backend of the target model",
    )

    # dataset arguments
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--train-hidden-states-path", type=str, default=None)
    dataset_group.add_argument("--eval-hidden-states-path", type=str, default=None)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="llama3")
    dataset_group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )
    dataset_group.add_argument("--build-dataset-num-proc", type=int, default=8)

    # training hyper params
    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=10)
    training_group.add_argument(
        "--max-num-steps",
        type=int,
        default=None,
        help="The maximum number of steps to train. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=1e-4)
    training_group.add_argument("--max-length", type=int, default=2048)
    training_group.add_argument("--warmup-ratio", type=float, default=0.015)
    training_group.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    training_group.add_argument("--max-grad-norm", type=float, default=0.5)
    training_group.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )
    training_group.add_argument(
        "--loss-schedule-path",
        type=str,
        default=None,
        help="Path to YAML file with external loss position weights schedule. "
             "If not provided, uses default exponential decay.",
    )
    training_group.add_argument(
        "--loss-decay-base",
        type=float,
        default=DEFAULT_LOSS_DECAY_BASE,
        help=f"Base for exponential position loss decay (default: {DEFAULT_LOSS_DECAY_BASE}). "
             "Weight for position i is base^i. Only used when no --loss-schedule-path is provided.",
    )
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="directory includes the checkpoint to start training with",
    )
    training_group.add_argument("--eval-interval", type=int, default=5000)
    training_group.add_argument("--save-interval", type=int, default=5000)
    training_group.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log training metrics every N steps",
    )
    training_group.add_argument("--seed", type=int, default=0)
    training_group.add_argument("--draft-accumulation-steps", type=int, default=1)

    # data processing type
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )
    optimization_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        help="The attention backend for the draft model",
    )

    # other args
    other_group = parser.add_argument_group("others")
    other_group.add_argument("--cache-key", type=str, default=None)
    other_group.add_argument("--cache-dir", type=str, default="./cache")
    other_group.add_argument("--output-dir", type=str, required=True)
    other_group.add_argument("--verbose", action="store_true")
    other_group.add_argument(
        "--dist-timeout",
        type=int,
        default=120,  # Increased from 20 to 120 min for large dataset preprocessing
        help="Timeout for collective communication in minutes",
    )
    other_group.add_argument(
        "--model-download-dir",
        type=str,
        default=None,
        help="The directory to download the target model to",
    )

    # vlm related args
    vlm_group = parser.add_argument_group("vlm")
    vlm_group.add_argument(
        "--min-pixels", type=int, default=50176
    )  # 64*28*28 for qwen2.5-vl
    vlm_group.add_argument(
        "--max-pixels", type=int, default=802816
    )  # 1024*28*28 for qwen2.5-vl

    # profiling related args
    profiling_group = parser.add_argument_group("profiling")
    profiling_group.add_argument("--profile", action="store_true")
    profiling_group.add_argument("--profile-start-step", type=int, default=30)
    profiling_group.add_argument("--profile-num-steps", type=int, default=4)
    profiling_group.add_argument("--profile-record-shapes", action="store_true")

    # sglang target model backend related args
    sglang_group = parser.add_argument_group("sglang target model backend")
    SGLangBackendArgs.add_args(sglang_group)

    # tracker related args
    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    args = parser.parse_args()
    return parser, args


def build_tracker(args: Namespace, parser: ArgumentParser) -> Tracker:
    """
    Build the experiment tracker according to the report_to argument.

    Args:
        args: The arguments for the training script.
        parser: The parser for the training script.

    Returns:
        The experiment tracker.
    """
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")
    tracker = create_tracker(args, args.output_dir)
    return tracker


def build_target_model(
    args: Namespace, draft_model_config: AutoDraftModelConfig, is_online: bool = True
) -> Tuple[Union[Eagle3TargetModel, TargetHead], Optional[AutoProcessor]]:
    """
    Build the target model according to the arguments.

    Args:
        args: The arguments for the training script.
        draft_model_config: The draft model config.

    Returns:
        The target model.
    """
    if is_online:
        if (
            args.is_vlm
            and draft_model_config.target_model_type == "qwen2_5_vl"
            and args.tp_size == 1
        ):
            from transformers import Qwen2_5_VLForConditionalGeneration

            target_model = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path=args.target_model_path,
                    torch_dtype=torch.bfloat16,
                )
                .eval()
                .cuda()
            )
        else:
            if args.target_model_backend == "sglang":
                target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
            else:
                target_model_kwargs = {}
            target_model = get_eagle3_target_model(
                pretrained_model_name_or_path=args.target_model_path,
                backend=args.target_model_backend,
                torch_dtype=torch.bfloat16,
                device="cuda",
                cache_dir=args.model_download_dir,
                **target_model_kwargs,
            )

        # set the aux hidden states layers
        if (
            hasattr(draft_model_config, "eagle_config")
            and draft_model_config.eagle_config is not None
            and "eagle_aux_hidden_state_layer_ids" in draft_model_config.eagle_config
        ):
            target_model.set_aux_hidden_states_layers(
                draft_model_config.eagle_config["eagle_aux_hidden_state_layer_ids"]
            )
        else:
            target_model.set_aux_hidden_states_layers()

        if args.is_vlm:
            processor = AutoProcessor.from_pretrained(
                args.target_model_path,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
        else:
            processor = None

        return target_model, processor
    else:
        target_head = TargetHead.from_pretrained(
            model_path=args.target_model_path,
            lm_head_key=args.lm_head_key,
            cache_dir=args.model_download_dir,
        )
        return target_head, None


def sanity_check(args: Namespace) -> None:
    """
    Perform sanity checks on the arguments.

    Args:
        args: The arguments for the training script.

    Returns:
        None
    """
    args.dp_size = dist.get_world_size() // args.tp_size
    args.target_batch_size = args.tp_size * args.batch_size


def build_draft_model(
    args: Namespace,
) -> Tuple[AutoDraftModelConfig, nn.Module, Optional[str]]:
    """Build the draft model for EAGLE3 training.

    Args:
        args: The arguments for the training script.

    Returns:
        A tuple of:
        - draft_model_config: The configuration for the draft model
        - draft_model: The initialized draft model
        - resume_checkpoint_path: Path to checkpoint for training state restoration,
            or None if not resuming from a checkpoint with training state.
            Note: This is only set when --resume finds a checkpoint in output_dir,
            NOT when using --ckpt-dir (which is for finetuning from a base model).
    """
    # Handle draft model config
    if args.draft_model_config is None:
        # Auto-generate and save config file
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path, cache_dir=args.model_download_dir
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        # Use provided config file
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    # Handle base ckpt, config file
    # Note: --ckpt-dir is for finetuning from a pretrained draft model (e.g., Tengyunw)
    # It does NOT have training state (optimizer, scheduler, global_step) to restore
    draft_model_last_checkpoint = None
    resume_checkpoint_path = None  # Only set when resuming with training state

    if args.ckpt_dir is not None:
        if os.path.isdir(args.ckpt_dir):
            # FIX: load config as object, not string path (was causing .vocab_size crash)
            draft_model_config = AutoDraftModelConfig.from_file(
                os.path.join(args.ckpt_dir, "config.json")
            )
            draft_model_last_checkpoint = args.ckpt_dir
            print_on_rank0(f"Finetuning from base model: {draft_model_last_checkpoint}")
        else:
            raise ValueError(
                f"Provided base model dir {args.ckpt_dir} is not a valid directory."
            )

    # Detecting last ckpt for draft model when resuming
    # This is the checkpoint that has training state (optimizer, scheduler, global_step)
    if args.resume and os.path.isdir(args.output_dir):
        if dist.get_rank() == 0:
            print(f"[RESUME] Checking for checkpoints in: {args.output_dir}", flush=True)
        found_checkpoint = get_last_checkpoint(args.output_dir)
        if found_checkpoint:
            draft_model_last_checkpoint = found_checkpoint
            resume_checkpoint_path = found_checkpoint  # Mark this for training state restoration
            if dist.get_rank() == 0:
                print(f"[RESUME] ✓ Checkpoint detected: {resume_checkpoint_path}", flush=True)
        else:
            if dist.get_rank() == 0:
                print("[RESUME] ✗ No valid checkpoint found, starting fresh", flush=True)
    elif args.resume:
        if dist.get_rank() == 0:
            print(f"[RESUME] Output dir does not exist yet: {args.output_dir}", flush=True)

    if draft_model_last_checkpoint:
        draft_model = AutoEagle3DraftModel.from_pretrained(
            draft_model_last_checkpoint,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        draft_model = AutoEagle3DraftModel.from_config(
            draft_model_config,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()

    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    return draft_model_config, draft_model, resume_checkpoint_path


def build_dataloaders(
    args: Namespace,
    draft_model_config: AutoDraftModelConfig,
    processor: Optional[AutoProcessor] = None,
) -> Tuple[DataLoader, str, Optional[DataLoader]]:
    """Build train and eval dataloaders.

    Supports two data loading paths:
    1. FAST PATH: Pre-processed Arrow cache (from preprocess_specforge_data.py)
       - Detects arrow_cache/ and vocab_mapping.pt in train_data_path directory
       - Loads instantly via load_from_disk()
       - No rank_0_priority() needed, all ranks load in parallel

    2. SLOW PATH: Raw JSON files (original behavior)
       - Loads JSON, tokenizes, builds vocab mapping at training time
       - All done on rank 0 while other ranks wait at barrier
       - Can cause NCCL timeouts for large datasets
    """
    from datasets import load_from_disk

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # Check for pre-processed Arrow cache (FAST PATH)
    # Expected structure: /path/to/data/arrow_cache/ and /path/to/data/vocab_mapping.pt
    # logic to find the directory, handling both direct paths and glob patterns
    if os.path.isdir(args.train_data_path):
        train_data_dir = args.train_data_path.rstrip("/")
    else:
        # If path is a glob (e.g. /path/to/*.jsonl) or file, try to find the parent directory
        # Split at first wildcard if present to get the base directory
        clean_path = args.train_data_path.split("*")[0].rstrip("/")
        if os.path.isdir(clean_path):
            train_data_dir = clean_path
        else:
            # If it's a file path or non-existent dir, take dirname
            train_data_dir = os.path.dirname(clean_path)

    # Check candidates for FAST PATH
    # Candidate 1: train_data_dir is the parent of arrow_cache (Standard Preprocess Output)
    c1_arrow = os.path.join(train_data_dir, "arrow_cache")
    c1_vocab = os.path.join(train_data_dir, "vocab_mapping.pt")

    # Candidate 2: train_data_dir IS the arrow_cache directory (User pointed deep)
    # (vocab mapping should be in parent)
    c2_arrow = train_data_dir
    c2_vocab = os.path.join(os.path.dirname(train_data_dir), "vocab_mapping.pt")

    is_fast_path = False
    if os.path.isdir(c1_arrow) and os.path.isfile(c1_vocab):
        arrow_cache_dir = c1_arrow
        vocab_mapping_path_candidate = c1_vocab
        is_fast_path = True
    elif os.path.basename(c2_arrow) == "arrow_cache" and os.path.isdir(c2_arrow) and os.path.isfile(c2_vocab):
        arrow_cache_dir = c2_arrow
        vocab_mapping_path_candidate = c2_vocab
        is_fast_path = True

    if is_fast_path:
        # FAST PATH: Load pre-processed Arrow cache
        print_on_rank0(f"[FAST PATH] Loading pre-processed Arrow cache from {arrow_cache_dir}")
        print_on_rank0(f"[FAST PATH] Using pre-computed vocab mapping from {vocab_mapping_path_candidate}")

        # Validate vocab sizes match (if metadata exists)
        metadata_path = os.path.join(os.path.dirname(arrow_cache_dir), "dataset_metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            expected_draft_vocab = metadata.get("draft_vocab_size")
            expected_target_vocab = metadata.get("target_vocab_size")
            if expected_draft_vocab and expected_draft_vocab != draft_model_config.draft_vocab_size:
                raise ValueError(
                    f"[FAST PATH] Vocab size mismatch! Preprocessed with draft_vocab_size={expected_draft_vocab}, "
                    f"but draft_model_config expects {draft_model_config.draft_vocab_size}. "
                    f"Re-run preprocessing with correct vocab sizes."
                )
            if expected_target_vocab and expected_target_vocab != draft_model_config.vocab_size:
                raise ValueError(
                    f"[FAST PATH] Vocab size mismatch! Preprocessed with target_vocab_size={expected_target_vocab}, "
                    f"but draft_model_config expects {draft_model_config.vocab_size}. "
                    f"Re-run preprocessing with correct vocab sizes."
                )
            print_on_rank0(f"[FAST PATH] Vocab sizes validated: draft={expected_draft_vocab}, target={expected_target_vocab}")

        train_eagle3_dataset = load_from_disk(arrow_cache_dir)
        train_eagle3_dataset.set_format(type="torch")
        vocab_mapping_path = vocab_mapping_path_candidate

        print_on_rank0(f"[FAST PATH] Loaded {len(train_eagle3_dataset):,} samples instantly!")

        # Handle offline training (hidden states) if specified
        if args.train_hidden_states_path is not None:
            train_eagle3_dataset = build_offline_eagle3_dataset(
                args.train_hidden_states_path,
                args.max_length,
            )
    else:
        # SLOW PATH: Process from JSON files (original behavior)
        print_on_rank0("[SLOW PATH] No Arrow cache found, processing from JSON files")
        print_on_rank0("[SLOW PATH] To speed this up, run preprocess_specforge_data.py first")

        cache_params_string = (
            f"{args.train_data_path}-"
            f"{args.max_length}-"
            f"{args.chat_template}-"
            f"{args.target_model_path}"
        )
        cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
        train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
        with rank_0_priority():
            train_eagle3_dataset = build_eagle3_dataset(
                dataset=train_dataset,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
                cache_key=cache_key,
                is_vlm=args.is_vlm,
                is_preformatted=args.is_preformatted,
                processor=processor,
                num_proc=args.build_dataset_num_proc,
            )
            vocab_mapping_path = generate_vocab_mapping_file(
                dataset=train_eagle3_dataset,
                target_vocab_size=draft_model_config.vocab_size,
                draft_vocab_size=draft_model_config.draft_vocab_size,
                cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
                cache_key=cache_key,
                num_proc=args.build_dataset_num_proc,
            )

            if args.train_hidden_states_path is not None:
                train_eagle3_dataset = build_offline_eagle3_dataset(
                    args.train_hidden_states_path,
                    args.max_length,
                )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.target_batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=args.is_vlm,
        seed=args.seed,  # Ensure reproducible shuffling across restarts
    )

    if args.eval_data_path is not None or args.eval_hidden_states_path is not None:
        if args.eval_data_path is not None:
            eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
            eval_eagle3_dataset = build_eagle3_dataset(
                eval_dataset,
                tokenizer,
                args.chat_template,
                args.max_length,
                is_vlm=args.is_vlm,
                processor=processor,
                num_proc=args.build_dataset_num_proc,
                is_preformatted=args.is_preformatted,
            )
        elif args.eval_hidden_states_path is not None:
            eval_eagle3_dataset = build_offline_eagle3_dataset(
                args.eval_hidden_states_path,
                args.max_length,
            )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.target_batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
            is_vlm=args.is_vlm,
            seed=args.seed,  # Consistent seeding with train dataloader
        )
        print_with_rank("Initialized eval dataloader")
    else:
        eval_dataloader = None
    return (
        train_dataloader,
        vocab_mapping_path,
        eval_dataloader,
    )


def save_checkpoints(
    args: Namespace,
    epoch: int,
    step: int,
    eagle3_model: nn.Module,
    optimizer: Optimizer,
    step_in_epoch: int = 0,  # Track position within epoch for intra-epoch resume
    loss_schedule: Optional[ExternalSchedule] = None,  # NEW: External loss schedule
):
    epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(epoch_output_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(eagle3_model, StateDictType.FULL_STATE_DICT):
        model_state_dict = eagle3_model.state_dict()
        state_to_save = {
            "epoch": epoch,
            "global_step": step,
            "step_in_epoch": step_in_epoch,  # For intra-epoch resume
            "args": args,
        }
        # NEW: Save schedule state for checkpoint validation on resume
        if loss_schedule is not None:
            state_to_save["loss_schedule"] = loss_schedule.to_checkpoint_state()
        state_to_save.update(optimizer.state_dict())
        draft_model_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in model_state_dict.items()
            if "draft_model." in k and "embed" not in k.lower()
        }

        if dist.get_rank() == 0:
            torch.save(
                state_to_save,
                os.path.join(epoch_output_dir, "training_state.pt"),
            )
            print_on_rank0(
                f"Saved full training state to {epoch_output_dir}/training_state.pt"
            )
            eagle3_model.draft_model.save_pretrained(
                epoch_output_dir,
                state_dict=draft_model_state_dict,
            )
            print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
        dist.barrier()


def run_forward(
    args: Namespace,
    eagle3_model: nn.Module,
    data: dict,
    target_model: Optional[Eagle3TargetModel] = None,
    is_online: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if args.is_vlm:
        plosses, _, acces = eagle3_model(
            input_ids=data["input_ids"].cuda(),
            attention_mask=data["attention_mask"].cuda(),
            loss_mask=data["loss_mask"].cuda(),
            pixel_values=data["pixel_values"].cuda(),
            image_grid_thw=data["image_grid_thw"].cuda(),
        )
    else:
        if is_online:
            # we generate the eagle3 using the target model in an online fashion
            eagle3_data = target_model.generate_eagle3_data(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )

            input_ids = get_dp_data_shard_from_tp(eagle3_data.input_ids)
            attention_mask = get_dp_data_shard_from_tp(eagle3_data.attention_mask)
            loss_mask = get_dp_data_shard_from_tp(eagle3_data.loss_mask)
            target = get_dp_data_shard_from_tp(eagle3_data.target)
            hidden_states = get_dp_data_shard_from_tp(eagle3_data.hidden_states)
        else:
            # we generate the logits using the hidden states loaded from disk
            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            hidden_states = data["hidden_state"].cuda()
            target = target_model(data["target"].cuda())
            input_ids, target, loss_mask = target_model.preprocess(
                input_ids, target, loss_mask
            )

        plosses, _, acces = eagle3_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target=target,
            hidden_states=hidden_states,
        )
    return plosses, acces


def run_backward_and_update(
    args: Namespace,
    plosses: List[torch.Tensor],
    optimizer: Optimizer,
    global_step: int,
    loss_schedule: Optional["ExternalSchedule"] = None,  # NEW: External schedule for loss weights
) -> None:
    """Apply position-weighted loss and update optimizer.

    Args:
        args: Training arguments
        plosses: Per-position losses from the model
        optimizer: The optimizer
        global_step: Current global training step
        loss_schedule: External schedule for dynamic loss weights.
                      Should always be provided (use ExternalSchedule.default() if none specified).
    """
    # Get loss weights from schedule (or use default constant)
    if loss_schedule is not None:
        ploss_weight = loss_schedule.get_loss_weights(global_step, len(plosses))
    else:
        # Fallback: use default constant (shouldn't happen if caller uses ExternalSchedule.default())
        ploss_weight = [DEFAULT_LOSS_DECAY_BASE ** i for i in range(len(plosses))]

    ploss = (
        sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        / args.draft_accumulation_steps
    )
    ploss.backward()

    if global_step % args.draft_accumulation_steps == 0:
        optimizer.step()


def record_metrcs(
    args: Namespace,
    accuracies: List[torch.Tensor],
    plosses: List[torch.Tensor],
    global_step: int,
    tracker: Tracker,
    optimizer: Optional[Optimizer] = None,
    mode: str = "train",
    source_ids: Optional[torch.Tensor] = None,
    source_registry: Optional[Dict[str, str]] = None,
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    accuracies = torch.stack(accuracies)
    plosses = torch.stack(plosses)

    assert accuracies.shape[0] == args.ttt_length
    dist.all_reduce(accuracies, op=dist.ReduceOp.AVG)
    accuracies = accuracies.cpu().tolist()
    for i in range(len(accuracies)):
        logdict[f"{mode}/acc_{i}"] = accuracies[i]
        print_on_rank0(
            f"Eval - Step {global_step} [{global_step + 1}/{args.num_epochs}], position {i},  Acc: {accuracies[i]:.2f}"
        )

    dist.all_reduce(plosses, op=dist.ReduceOp.AVG)
    plosses = plosses.cpu().tolist()
    for i in range(len(plosses)):
        logdict[f"{mode}/ploss_{i}"] = plosses[i]
        print_on_rank0(
            f"Eval - Step {global_step} [{global_step + 1}/{args.num_epochs}], position {i}, pLoss: {plosses[i]}"
        )

    # Log per-source distribution if available (with meaningful names)
    if source_ids is not None and source_ids.numel() > 0:
        unique_sources, counts = source_ids.unique(return_counts=True)
        total_samples = source_ids.numel()
        registry = source_registry or {}
        for sid, cnt in zip(unique_sources.tolist(), counts.tolist()):
            # Use source name from registry if available, otherwise fall back to id
            source_name = registry.get(str(sid), str(sid))
            logdict[f"{mode}/source_ratio/{source_name}"] = cnt / total_samples

    tracker.log(logdict, step=global_step)


def get_dp_data_shard_from_tp(tensor: torch.Tensor) -> torch.Tensor:
    """
    Get the data shard from the tensor.
    """
    tp_size = dist.get_world_size(get_tp_group())
    tp_rank = dist.get_rank(get_tp_group())
    return tensor.chunk(tp_size, dim=0)[tp_rank]


def main():
    # ================================================
    # 1. Initialize
    # ================================================
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    is_online = (
        args.train_data_path is not None and args.train_hidden_states_path is None
    )

    sanity_check(args)
    print_with_rank("Initialized distributed environment")

    # ================================================
    # 1.5 Load External Schedule (if provided)
    # ================================================
    if args.loss_schedule_path:
        loss_schedule = ExternalSchedule.from_file(args.loss_schedule_path, default_base=args.loss_decay_base)
        print_on_rank0(f"[SCHEDULE] Loaded loss schedule from {args.loss_schedule_path}")
        print_on_rank0(f"[SCHEDULE] Description: {loss_schedule.description}")
        print_on_rank0(f"[SCHEDULE] Checksum: {loss_schedule.checksum}")
    else:
        loss_schedule = ExternalSchedule.default(default_base=args.loss_decay_base)
        print_on_rank0(f"[SCHEDULE] Using default exponential decay ({args.loss_decay_base}^i)")

    # ================================================
    # 2. Build models
    # ================================================
    draft_model_config, draft_model, resume_checkpoint_path = build_draft_model(args)
    target_model, processor = build_target_model(args, draft_model_config, is_online)

    # ================================================
    # 3. Build dataloader
    # ================================================
    train_dataloader, vocab_mapping_path, eval_dataloader = build_dataloaders(
        args, draft_model_config, processor
    )

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    # Load source registry for meaningful per-source logging
    source_registry = {}
    metadata_paths = [
        os.path.join(args.train_data_path, "dataset_metadata.json"),
        os.path.join(os.path.dirname(args.train_data_path), "dataset_metadata.json"),
    ]
    for metadata_path in metadata_paths:
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            source_registry = metadata.get("source_registry", {})
            if source_registry:
                print_on_rank0(f"[SOURCE] Loaded source registry: {source_registry}")
            break

    # Calculate total steps if not provided
    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch
        print_with_rank(
            f"Auto-calculated total_steps: {args.total_steps} (num_epochs={args.num_epochs} * steps_per_epoch={steps_per_epoch})"
        )
    else:
        print_with_rank(f"Using provided total_steps: {args.total_steps}")

    # ================================================
    # 4. Build Eagle3 model
    # ================================================
    if (
        args.is_vlm
        and getattr(draft_model_config, "target_model_type", None) == "qwen2_5_vl"
    ):
        eagle3_model = QwenVLOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    else:
        eagle3_model = OnlineEagle3Model(
            draft_model=draft_model,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )

    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        process_group=dist.group.WORLD,  # the draft model should run dp for all processes
    )
    print_with_rank("Initialized Eagle3 FSDP model")

    # ================================================
    # 5. Build optimizer and scheduler
    # ================================================
    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
    )
    print_with_rank("Initialized optimizer and scheduler")

    # ================================================
    # 5.1 Restore training state if resuming
    # ================================================
    # Initialize defaults - will be overwritten if resuming
    global_step = 0
    start_epoch = 0
    step_in_epoch_resume = 0  # NEW: For intra-epoch resume (skip already-processed batches)

    if resume_checkpoint_path is not None:
        training_state_path = os.path.join(resume_checkpoint_path, "training_state.pt")
        if os.path.isfile(training_state_path):
            if dist.get_rank() == 0:
                print(f"[RESUME] Loading training state from: {training_state_path}", flush=True)
            # Load on CPU first to avoid GPU memory fragmentation
            # weights_only=False required because training_state contains argparse.Namespace
            # This is safe since we're loading our own checkpoint files
            training_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )

            # Restore optimizer and scheduler states
            # The training_state contains optimizer_state_dict and scheduler_state_dict
            # from optimizer.state_dict() call in save_checkpoints
            optimizer.load_state_dict(training_state)

            # Restore training progress
            global_step = training_state.get("global_step", 0)
            start_epoch = training_state.get("epoch", 0)
            # NEW: Restore intra-epoch position for proper resume (avoids reprocessing batches)
            step_in_epoch_resume = training_state.get("step_in_epoch", 0)

            if dist.get_rank() == 0:
                print(
                    f"[RESUME] ✓ Restored: epoch={start_epoch}, global_step={global_step}, "
                    f"step_in_epoch={step_in_epoch_resume}, lr={optimizer.get_learning_rate():.2e}",
                    flush=True,
                )
        else:
            if dist.get_rank() == 0:
                print(
                    f"[RESUME] ✗ training_state.pt not found at {training_state_path}, "
                    "optimizer state lost",
                    flush=True,
                )

    # ================================================
    # 6. Build tracker
    # ================================================
    tracker = build_tracker(args, parser)
    dist.barrier()

    last_time = time.time()

    # ================================================
    # 7. Start training
    # ================================================
    if dist.get_rank() == 0:
        print(
            f"[TRAIN] Starting from epoch={start_epoch}, global_step={global_step}",
            flush=True,
        )

    for epoch in range(start_epoch, args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()

        # Log skip info if resuming mid-epoch
        if epoch == start_epoch and step_in_epoch_resume > 0:
            if dist.get_rank() == 0:
                print(
                    f"[RESUME] Skipping {step_in_epoch_resume} batches in epoch {epoch} "
                    f"(already processed before checkpoint)",
                    flush=True,
                )

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                enumerate(train_dataloader),
                desc=f"Training Epoch {epoch}",
                leave=True,
                total=len(train_dataloader),
            )
        else:
            progress_bar = enumerate(train_dataloader)

        # Track current position in epoch for checkpointing
        current_step_in_epoch = 0

        for batch_idx, data in progress_bar:
            # INTRA-EPOCH RESUME: Skip already-processed batches
            # This is safe because DistributedSampler is deterministic with same seed + epoch
            if epoch == start_epoch and batch_idx < step_in_epoch_resume:
                continue

            current_step_in_epoch = batch_idx + 1  # Track position (1-indexed for next resume)
            global_step += 1

            # ================================================
            # 7.0 Profiling
            # ================================================
            if args.profile:
                # we add the step by 1 to align with global step
                if global_step == args.profile_start_step + 1:
                    print("Start profile")
                    torch_profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        record_shapes=args.profile_record_shapes,
                    )
                    torch_profiler.start()
                if global_step == args.profile_start_step + args.profile_num_steps + 1:
                    output_path = os.path.join(
                        args.output_dir,
                        f"profile_rank{torch.distributed.get_rank()}_{time.time()}.trace.json.gz",
                    )
                    print(f"End profile {output_path=}")
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(output_path)

            # ================================================
            # 7.1 Training Step
            # ================================================
            # Extract source_id for per-source logging (if present in batch)
            batch_source_ids = data.pop("source_id", None)

            plosses, acces = run_forward(
                args, eagle3_model, data, target_model, is_online
            )
            run_backward_and_update(args, plosses, optimizer, global_step, loss_schedule)

            # log training metrics
            if global_step % args.log_interval == 0:
                record_metrcs(
                    args, acces, plosses, global_step, tracker, optimizer, mode="train",
                    source_ids=batch_source_ids,
                    source_registry=source_registry,
                )

            if dist.get_rank() == 0:
                time_per_step = time.time() - last_time
                last_time = time.time()
                avg_loss = sum(pl for pl in plosses) / len(plosses)
                avg_acc = sum(acces) / len(acces)
                progress_bar.set_postfix(
                    {
                        "step": global_step,
                        "loss": f"{avg_loss:.2f}",
                        "acc": f"{avg_acc:.2f}",
                        "time": f"{time_per_step:.2f}s",
                    }
                )

            # ================================================
            # 7.2 Evaluation Step
            # ================================================
            if (
                args.eval_data_path is not None
                and global_step % args.eval_interval == 0
            ):
                # Run evaluation
                draft_model.eval()
                eval_acces = [[] for _ in range(eagle3_model.length)]
                eval_plosses = [[] for _ in range(eagle3_model.length)]
                eval_source_ids = []  # Collect source_ids across all eval batches

                for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                    # Extract source_id for per-source logging (if present in batch)
                    eval_batch_source_ids = data.pop("source_id", None)
                    if eval_batch_source_ids is not None:
                        eval_source_ids.append(eval_batch_source_ids)

                    with torch.no_grad():
                        plosses, acces = run_forward(
                            args, eagle3_model, data, target_model, is_online
                        )
                        eval_acces = [
                            eval_acces[i] + [acces[i]] for i in range(len(acces))
                        ]
                        eval_plosses = [
                            eval_plosses[i] + [plosses[i]] for i in range(len(plosses))
                        ]

                # compute average over all minibatches
                eval_acces = [torch.stack(acc).mean() for acc in eval_acces]
                eval_plosses = [torch.stack(pl).mean() for pl in eval_plosses]

                # Concatenate all eval source_ids for distribution logging
                all_eval_source_ids = torch.cat(eval_source_ids) if eval_source_ids else None

                record_metrcs(
                    args,
                    eval_acces,
                    eval_plosses,
                    global_step,
                    tracker,
                    mode="eval",
                    source_ids=all_eval_source_ids,
                    source_registry=source_registry,
                )

            # ================================================
            # 7.3 Save Checkpoints
            # ================================================
            if global_step % args.save_interval == 0:
                # Save the model with intra-epoch position and schedule state for proper resume
                save_checkpoints(
                    args, epoch, global_step, eagle3_model, optimizer,
                    step_in_epoch=current_step_in_epoch,
                    loss_schedule=loss_schedule,
                )

            if args.max_num_steps is not None and global_step >= args.max_num_steps:
                break

        # Reset intra-epoch skip after first resumed epoch completes
        # (subsequent epochs start from batch 0)
        if epoch == start_epoch and step_in_epoch_resume > 0:
            if dist.get_rank() == 0:
                print(f"[RESUME] Epoch {epoch} complete, skip disabled for subsequent epochs", flush=True)
            step_in_epoch_resume = 0

        if args.max_num_steps is not None and global_step >= args.max_num_steps:
            break

    # Close the tracker
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
