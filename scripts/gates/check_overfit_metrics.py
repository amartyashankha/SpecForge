"""Gate an overfit run on its final loss, accuracy, and checkpoint.

Assumptions about the trainer log and output structure:

- Metric lines use either the historical disaggregated consumer form
  ``[consumer] step <N> {<dict>}`` or the unified trainer form
  ``step <N>: {<dict>}``.
- The checkpoint tree layout is:
  ``<checkpoint_root>/<run-id>-step<N>/training_state.pt``
  If a new method saves checkpoints under a different layout, adjust
  ``checkpoint_paths()`` to match.
- The gate thresholds ``--max-loss`` and ``--min-accuracy`` are method-agnostic
  CLI arguments; every future overfit gate can pass the values appropriate for that
  method without modifying this script.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple


def _metric_line(line: str):
    """Parse one exact trainer metric line without matching arbitrary log text."""

    value = line.strip()
    historical_prefix = "[consumer] step "
    unified_prefix = "step "
    if value.startswith(historical_prefix):
        remainder = value[len(historical_prefix) :]
        step_text, separator, metrics_text = remainder.partition(" ")
    elif value.startswith(unified_prefix):
        remainder = value[len(unified_prefix) :]
        step_text, separator, metrics_text = remainder.partition(":")
    else:
        return None
    if not separator or not step_text.isdigit():
        return None
    metrics = ast.literal_eval(metrics_text.strip())
    if not isinstance(metrics, dict):
        raise ValueError(f"step {step_text} metrics must be a dict")
    return int(step_text), metrics


def metric_history(log_path: str):
    matches = []
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parsed = _metric_line(line)
            if parsed is not None:
                matches.append(parsed)
    if not matches:
        raise ValueError(f"no consumer metric lines found in {log_path}")
    return matches


def final_metrics(log_path: str) -> Tuple[int, Dict[str, float]]:
    return metric_history(log_path)[-1]


def checkpoint_paths(checkpoint_root: str):
    paths = list(Path(checkpoint_root).glob("*-step*/training_state.pt"))

    def get_step(path: Path) -> int:
        _, separator, suffix = path.parent.name.rpartition("-step")
        return int(suffix) if separator and suffix.isdigit() else -1

    return [(get_step(path), str(path)) for path in sorted(paths, key=get_step)]


def _aliased_scalar(
    metrics: Dict[str, float], *, names: Tuple[str, ...], label: str
) -> Tuple[str, float]:
    present = [name for name in names if name in metrics]
    if not present:
        raise ValueError(
            f"final metrics contain no {label}; accepted keys are {list(names)}"
        )
    values = {name: float(metrics[name]) for name in present}
    if len(set(values.values())) != 1:
        raise ValueError(
            f"final metrics disagree between {label} aliases: {values}"
        )
    key = next(name for name in names if name in values)
    return key, values[key]


def _loss(metrics: Dict[str, float]) -> Tuple[str, float]:
    return _aliased_scalar(
        metrics,
        names=("loss", "train/loss"),
        label="loss",
    )


def _accuracy(metrics: Dict[str, float]) -> Tuple[str, float]:
    return _aliased_scalar(
        metrics,
        names=("acc", "train/acc", "accuracy", "train/accuracy"),
        label="token accuracy",
    )


def _position_accuracies(metrics: Dict[str, float]) -> Dict[str, float]:
    values = {}
    for name, raw in metrics.items():
        canonical = str(name)
        if canonical.startswith("train/"):
            canonical = canonical[len("train/") :]
        prefix, separator, suffix = canonical.partition("_")
        if prefix == "acc" and separator and suffix.isdigit():
            if canonical in values and values[canonical] != float(raw):
                raise ValueError(
                    f"final metrics disagree between aliases for {canonical}"
                )
            values[canonical] = float(raw)
    return dict(
        sorted(values.items(), key=lambda item: int(item[0].partition("_")[2]))
    )


def check_overfit(
    log_path: str,
    checkpoint_root: str,
    *,
    expected_step: int,
    expected_first_step: Optional[int] = None,
    max_loss: float,
    min_accuracy: float,
    require_position_accuracy: bool = False,
    expected_position_count: Optional[int] = None,
) -> Dict[str, object]:
    history = metric_history(log_path)
    first_step = history[0][0]
    step, metrics = history[-1]
    loss_key, loss = _loss(metrics)
    accuracy_key, accuracy = _accuracy(metrics)
    position_accuracies = _position_accuracies(metrics)
    checkpoints = checkpoint_paths(checkpoint_root)
    errors = []
    if expected_first_step is not None and first_step != expected_first_step:
        errors.append(
            f"first logged step {first_step} != expected {expected_first_step}"
        )
    if step != expected_step:
        errors.append(f"final logged step {step} != expected {expected_step}")
    if not math.isfinite(loss):
        errors.append(f"final loss is not finite: {loss}")
    elif loss > max_loss:
        errors.append(f"final loss {loss} > {max_loss}")
    if not math.isfinite(accuracy):
        errors.append(f"final token accuracy is not finite: {accuracy}")
    elif accuracy < min_accuracy:
        errors.append(f"final token accuracy {accuracy} < {min_accuracy}")
    if expected_position_count is not None and expected_position_count < 1:
        errors.append("expected_position_count must be at least 1")
    positions_required = (
        require_position_accuracy or expected_position_count is not None
    )
    if positions_required and not position_accuracies:
        errors.append("final metrics contain no per-position acc_<N> values")
    if expected_position_count is not None:
        expected_positions = {
            f"acc_{index}" for index in range(expected_position_count)
        }
        actual_positions = set(position_accuracies)
        if actual_positions != expected_positions:
            errors.append(
                "per-position accuracy keys differ from expected: "
                f"missing={sorted(expected_positions - actual_positions)}, "
                f"extra={sorted(actual_positions - expected_positions)}"
            )
    for name, value in position_accuracies.items():
        if not math.isfinite(value):
            errors.append(f"final {name} is not finite: {value}")
        elif positions_required and value < min_accuracy:
            errors.append(f"final {name} {value} < {min_accuracy}")
    if not checkpoints:
        errors.append(f"no training_state.pt checkpoint under {checkpoint_root}")
        checkpoint_step = checkpoint = None
    else:
        checkpoint_step, checkpoint = checkpoints[-1]
        if checkpoint_step != expected_step:
            errors.append(
                f"final checkpoint step {checkpoint_step} != expected {expected_step}"
            )
    result = {
        "first_step": first_step,
        "step": step,
        "loss": loss,
        "loss_key": loss_key,
        "token_accuracy": accuracy,
        "accuracy_key": accuracy_key,
        "position_accuracies": position_accuracies,
        "checkpoint_step": checkpoint_step,
        "checkpoint": checkpoint,
        "passed": not errors,
    }
    if errors:
        raise ValueError("; ".join(errors))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument(
        "--expected-first-step",
        type=int,
        default=None,
        help="require the first metric step, e.g. the first post-resume step",
    )
    parser.add_argument("--max-loss", type=float, default=1e-4)
    parser.add_argument("--min-accuracy", type=float, default=1.0)
    parser.add_argument(
        "--require-position-accuracy",
        action="store_true",
        help="require every logged EAGLE3 acc_<N> value to meet --min-accuracy",
    )
    parser.add_argument(
        "--expected-position-count",
        type=int,
        default=None,
        help="require exactly acc_0 through acc_<N-1> (implies position checking)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = check_overfit(
        args.log_path,
        args.checkpoint_root,
        expected_step=args.expected_step,
        expected_first_step=args.expected_first_step,
        max_loss=args.max_loss,
        min_accuracy=args.min_accuracy,
        require_position_accuracy=args.require_position_accuracy,
        expected_position_count=args.expected_position_count,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
