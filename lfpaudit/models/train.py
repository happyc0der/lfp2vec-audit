"""The fine-tuning loop.

Written by hand rather than delegated to a framework trainer, for three reasons that matter to
this project. Early stopping has to watch a held-out *group*, not a random validation slice, or
the stopping decision leaks across sessions. Every run has to write its manifest before the first
optimiser step, so an interrupted two-hour run still leaves evidence of what was attempted. And
the upstream entry point calls an experiment tracker unconditionally, which would make a
reproduction depend on having an account somewhere.

Hyper-parameters follow the paper where it states them: AdamW at 3e-5, effective batch 32 reached
by accumulation, ten epochs, warmup over the first tenth. Early stopping is added because a
pretrained encoder on twenty thousand chunks converges well before ten epochs, and on a budget
measured in evenings an hour spent overfitting is an hour not spent on another fold.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lfpaudit import REGIONS
from lfpaudit.eval.metrics import evaluate, softmax


@dataclass
class FineTuneConfig:
    """Everything that defines a run. Serialised into the manifest verbatim."""

    model_name: str = "facebook/wav2vec2-base"
    learning_rate: float = 3e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_per_class: int = 6000
    max_val_chunks: int = 4000
    freeze_feature_encoder: bool = True
    freeze_layers: int = 0
    lowpass_hz: float | None = None
    patience: int = 2
    seed: int = 0
    num_workers: int = 0

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class EpochRecord:
    """One epoch's worth of history, appended to a JSONL as it happens."""

    epoch: int
    train_loss: float
    val_balanced_accuracy: float
    val_loss: float
    seconds: float
    learning_rate: float
    improved: bool = False
    extra: dict = field(default_factory=dict)


def _loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def _linear_warmup_decay(step: int, total: int, warmup: int) -> float:
    """Linear warmup then linear decay, the schedule the paper's settings imply."""
    if warmup > 0 and step < warmup:
        return step / max(warmup, 1)
    remaining = max(total - warmup, 1)
    return max(0.0, (total - step) / remaining)


def predict(model, dataset, device: str, batch_size: int, num_workers: int = 0):
    """Logits and labels for a dataset, in order."""
    import torch

    model.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    with torch.no_grad():
        for waveforms, labels in _loader(dataset, batch_size, False, num_workers, 0):
            output = model(waveforms.to(device)).logits
            logits_out.append(output.float().cpu().numpy())
            labels_out.append(labels.numpy())
    return np.concatenate(logits_out), np.concatenate(labels_out)


def embed(model, dataset, device: str, batch_size: int, num_workers: int = 0) -> np.ndarray:
    """Pooled penultimate representations, for the lab-identity probe."""
    from lfpaudit.models.lfp2vec_lite import pooled_embeddings

    model.eval()
    out = [
        pooled_embeddings(model, waveforms.to(device))
        for waveforms, _ in _loader(dataset, batch_size, False, num_workers, 0)
    ]
    return np.concatenate(out)


def predict_with_embeddings(model, dataset, device: str, batch_size: int, num_workers: int = 0):
    """Logits, labels and pooled embeddings in one pass.

    Stage 3 saved logits but not embeddings, which left temperature scaling and any
    representation-based abstention unanswerable without retraining. Producing both together
    costs one forward pass instead of two and removes the reason that happened again.
    """
    import torch

    from lfpaudit.models.lfp2vec_lite import pooled_embeddings

    model.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    embeddings_out: list[np.ndarray] = []
    with torch.no_grad():
        for waveforms, labels in _loader(dataset, batch_size, False, num_workers, 0):
            batch = waveforms.to(device)
            logits_out.append(model(batch).logits.float().cpu().numpy())
            embeddings_out.append(pooled_embeddings(model, batch))
            labels_out.append(labels.numpy())
    return (
        np.concatenate(logits_out),
        np.concatenate(labels_out),
        np.concatenate(embeddings_out),
    )


def load_model(run_dir: str | Path, device: str = "cpu"):
    """Reload a fine-tuned model saved by a run, for post-hoc analysis."""
    from pathlib import Path as _Path

    from transformers import Wav2Vec2ForSequenceClassification

    path = _Path(run_dir) / "model"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; the run was made before --save-model, so its weights are gone"
        )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(path)
    model.eval()
    model.to(device)
    return model


def train(
    model,
    train_dataset,
    val_dataset,
    config: FineTuneConfig,
    device: str,
    run_dir: str | Path,
    log: bool = True,
) -> list[EpochRecord]:
    """Fine-tune, stopping on held-out-group balanced accuracy.

    Returns the per-epoch history. The best state by validation balanced accuracy is restored
    before returning, so the caller scores the model that was selected rather than the one that
    happened to exist when the loop ended.
    """
    import torch
    from torch.optim import AdamW

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"

    torch.manual_seed(config.seed)
    model.to(device)

    loader = _loader(train_dataset, config.batch_size, True, config.num_workers, config.seed)
    steps_per_epoch = max(1, len(loader) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(round(config.warmup_ratio * total_steps))

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: _linear_warmup_decay(step, total_steps, warmup_steps)
    )

    history: list[EpochRecord] = []
    best_score, best_state, since_improved = -np.inf, None, 0

    for epoch in range(config.epochs):
        started = time.time()
        model.train()
        running, batches = 0.0, 0
        optimiser.zero_grad(set_to_none=True)

        for index, (waveforms, labels) in enumerate(loader):
            output = model(waveforms.to(device), labels=labels.to(device))
            # Scaled so that accumulating N batches gives the same gradient as one large batch.
            (output.loss / config.gradient_accumulation_steps).backward()
            running += float(output.loss.detach().cpu())
            batches += 1

            if (index + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)

        val_logits, val_labels = predict(
            model, val_dataset, device, config.batch_size, config.num_workers
        )
        report = evaluate(softmax(val_logits), val_labels)
        score = report.balanced_accuracy
        improved = score > best_score

        record = EpochRecord(
            epoch=epoch,
            train_loss=running / max(batches, 1),
            val_balanced_accuracy=score,
            val_loss=report.nll,
            seconds=time.time() - started,
            learning_rate=float(scheduler.get_last_lr()[0]),
            improved=improved,
            extra={"val_chance": report.chance, "val_classes": report.classes_present},
        )
        history.append(record)
        with open(history_path, "a") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")
        if log:
            print(
                f"  epoch {epoch}: loss {record.train_loss:.4f} "
                f"val_bal_acc {score:.3f} (chance {report.chance:.3f}) "
                f"{record.seconds / 60:.1f} min{' *' if improved else ''}",
                flush=True,
            )

        if improved:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= config.patience:
                if log:
                    print(f"  stopping early after epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def smoke(device: str = "cpu", steps: int = 20, seed: int = 0) -> dict:
    """Train on synthetic separable data and assert the loop actually learns.

    This is the gate every real run passes first. It is not a test of the science; it is a test
    that gradients flow, that shapes line up, that the loss is finite, and that a model handed
    trivially separable input drives its loss down. A run that fails here would fail at hour two
    for the same reason.
    """
    import torch

    from lfpaudit.models.lfp2vec_lite import build_model

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n, samples = 16, 16000
    t = np.arange(samples) / 16000

    # Two classes that differ only in frequency, which the conv encoder can separate easily.
    labels = np.array([0, 4] * (n // 2))
    waveforms = np.stack(
        [np.sin(2 * np.pi * (5.0 if y == 0 else 60.0) * t) for y in labels]
    ) + 0.05 * rng.standard_normal((n, samples))
    waveforms = (waveforms - waveforms.mean(axis=1, keepdims=True)) / waveforms.std(
        axis=1, keepdims=True
    )

    model, info = build_model(num_labels=len(REGIONS))
    model.to(device).train()
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    inputs = torch.from_numpy(waveforms.astype(np.float32)).to(device)
    targets = torch.from_numpy(labels).to(device)

    losses = []
    for _ in range(steps):
        output = model(inputs, labels=targets)
        loss = output.loss
        if not torch.isfinite(loss):
            raise AssertionError("loss is not finite on the first steps")
        if output.logits.shape != (n, len(REGIONS)):
            raise AssertionError(f"unexpected logits shape {tuple(output.logits.shape)}")
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))

    if losses[-1] >= losses[0]:
        raise AssertionError(
            f"loss did not decrease over {steps} steps: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
    return {
        "device": device,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "steps": steps,
        "model": info.describe(),
    }


def measure_throughput(
    model, dataset, device: str, config: FineTuneConfig, steps: int = 20
) -> dict:
    """Time real training steps so a run can be sized before it is launched.

    The project's rule is that no expensive run starts on an estimate. This returns chunks per
    second under the actual configuration, which the caller turns into an epoch time and compares
    against the budget.
    """
    from torch.optim import AdamW

    model.to(device).train()
    optimiser = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
    loader = _loader(dataset, config.batch_size, True, config.num_workers, config.seed)

    seen, started = 0, None
    for index, (waveforms, labels) in enumerate(loader):
        if index == 2:  # discard the first batches, which pay one-off setup costs
            started = time.time()
            seen = 0
        output = model(waveforms.to(device), labels=labels.to(device))
        output.loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
        seen += len(labels)
        if started is not None and index >= steps + 1:
            break

    elapsed = time.time() - (started or time.time())
    rate = seen / max(elapsed, 1e-9)
    return {"chunks_per_second": rate, "batches_timed": steps, "device": device}
