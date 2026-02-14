import torch

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        max_consecutive_nan=100,
    ):
        # TODO: For now, we only support cosine annealing warmup lr scheduler and AdamW optimizer
        # TODO: We should make these parameters configurable
        #   These magic numbers: weight_decay=0.0, max_grad_norm=0.5, total_steps=800k, warmup_steps=12k are copied from
        #   https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [
            p.detach().clone().to(torch.float32) for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

        # NaN/Inf gradient tracking
        self._consecutive_nan_count = 0
        self._total_nan_count = 0
        self._max_consecutive_nan = max_consecutive_nan

    def step(self) -> bool:
        """Run one optimization step.

        Returns:
            True if the step was applied, False if skipped due to NaN/Inf gradients.

        Raises:
            RuntimeError: If NaN/Inf gradients persist for max_consecutive_nan steps.
        """
        # Copy BF16 grads to FP32 master params
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )

        # Clip gradients and check for NaN/Inf.
        # clip_grad_norm_ returns total_norm before clipping. If any grad contains
        # NaN/Inf, total_norm will be NaN/Inf (NaN propagates through torch.norm).
        # Note: With FSDP SHARD_GRAD_OP, gradients are all-reduced during backward,
        # so all ranks see identical grads and will agree on NaN detection without
        # explicit cross-rank synchronization. If you change to FULL_SHARD (which
        # shards gradients post-reduce), you'd need FSDP-aware norm computation.
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.fp32_params, self.max_grad_norm
        )

        if not torch.isfinite(total_norm):
            self._consecutive_nan_count += 1
            self._total_nan_count += 1

            # Log sparingly: first occurrence, then every 10th
            if (
                self._consecutive_nan_count == 1
                or self._consecutive_nan_count % 10 == 0
            ):
                print_on_rank0(
                    f"[BF16Optimizer] NaN/Inf gradient detected (total_norm={total_norm.item():.4f}). "
                    f"Skipping step. Consecutive: {self._consecutive_nan_count}, "
                    f"Total: {self._total_nan_count}"
                )

            if self._consecutive_nan_count >= self._max_consecutive_nan:
                raise RuntimeError(
                    f"[BF16Optimizer] {self._max_consecutive_nan} consecutive NaN/Inf "
                    f"gradients detected. Training is diverging. "
                    f"Total NaN steps: {self._total_nan_count}"
                )

            # Clean up FP32 grads (they are NaN after clip_grad_norm_ scaling)
            self.optimizer.zero_grad()
            # Clean up model grads — do NOT copy fp32 weights back (no update was applied)
            with torch.no_grad():
                for p in self.model_params:
                    p.grad = None

            # Skip both optimizer.step() AND scheduler.step(): the cosine schedule
            # should track effective optimization steps, not batches seen. Advancing
            # the scheduler on a skipped step causes premature LR decay.
            return False

        # Finite gradients — apply the update
        if self._consecutive_nan_count > 0:
            print_on_rank0(
                f"[BF16Optimizer] Gradients recovered after "
                f"{self._consecutive_nan_count} consecutive NaN steps "
                f"(total NaN: {self._total_nan_count})."
            )
        self._consecutive_nan_count = 0

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None
        return True

    def zero_grad(self):
        """Zero gradients on the bf16 model parameters.

        Used to discard partial gradient accumulation (e.g. at epoch boundaries)
        without triggering an optimizer step or scheduler advancement.
        """
        with torch.no_grad():
            for p in self.model_params:
                p.grad = None

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
