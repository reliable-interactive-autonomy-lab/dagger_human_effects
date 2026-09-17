"""Training loops: full pretraining and time-budgeted on-the-fly LoRA finetuning.

The finetuner is built around a wall-clock budget rather than a fixed step count,
because it runs *while a participant is waiting between demonstrations*. It also
supports a **sham** update (`commit=False`): the identical number of gradient steps
runs, so the compute and the perceived delay are unchanged, but the adapter weights
are rolled back afterwards. That is what makes the non-contingent study condition a
clean manipulation of controllability rather than a confounded change in timing.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PolicyConfig, TrainConfig
from .data import Episode, SequenceDataset, collate, compute_obs_stats
from .lora import (count_parameters, describe, inject_lora, lora_state_dict,
                   mark_trainable)
from .policies.base import BasePolicy


@dataclass
class TrainReport:
    steps: int = 0
    seconds: float = 0.0
    loss_start: float = float("nan")
    loss_end: float = float("nan")
    metrics: Dict[str, float] = field(default_factory=dict)
    committed: bool = True
    num_samples: int = 0
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps, "seconds": round(self.seconds, 3),
            "loss_start": self.loss_start, "loss_end": self.loss_end,
            "loss_delta": self.loss_end - self.loss_start,
            "committed": self.committed, "num_samples": self.num_samples,
            "note": self.note, **{f"m_{k}": v for k, v in self.metrics.items()},
        }


class PolicyTrainer:
    """Owns a policy plus its optimizer and knows how to (re)fit it on demos."""

    def __init__(self, policy: BasePolicy, train_cfg: TrainConfig,
                 policy_cfg: PolicyConfig, device: Optional[torch.device] = None):
        self.policy = policy
        self.tcfg = train_cfg
        self.pcfg = policy_cfg
        self.device = device or policy.device
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lora_active = False
        self._seq_lens = self._resolve_seq_lens()

    # ------------------------------------------------------------------ setup
    def _resolve_seq_lens(self) -> Dict[str, Any]:
        """Sequence windows and alignment mode required by the chosen policy."""
        if self.pcfg.algo == "act":
            # One observation in, a chunk of future actions out.
            return {"obs": 1, "action": int(self.pcfg.chunk_size),
                    "mode": SequenceDataset.CHUNKED}
        # BC-RNN consumes one observation per step and emits that step's action,
        # so observations and actions must span the same indices.
        return {"obs": int(self.pcfg.seq_length),
                "action": int(self.pcfg.seq_length),
                "mode": SequenceDataset.ALIGNED}

    def enable_lora(self) -> str:
        """Freeze the base policy and attach adapters. Idempotent."""
        if self.lora_active:
            return describe(self.policy)
        lc = self.pcfg.lora
        inject_lora(
            self.policy,
            rank=lc.rank, alpha=lc.alpha, dropout=lc.dropout,
            target_substrings=list(lc.target_substrings),
            adapt_rnn=lc.adapt_rnn,
        )
        mark_trainable(
            self.policy,
            train_biases=lc.train_biases,
            train_output_head=lc.train_output_head,
        )
        self.policy.to(self.device)
        self.lora_active = True
        self.optimizer = None  # parameter set changed
        return describe(self.policy)

    def _params(self) -> List[torch.nn.Parameter]:
        return [p for p in self.policy.parameters() if p.requires_grad]

    def _ensure_optimizer(self, lr: Optional[float] = None) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self._params(),
                lr=lr if lr is not None else (
                    self.tcfg.lr if self.lora_active else self.tcfg.base_lr),
                weight_decay=self.tcfg.weight_decay,
            )
        elif lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr
        return self.optimizer

    def reset_optimizer(self) -> None:
        """Drop Adam moment estimates (e.g. when the data distribution shifts a lot)."""
        self.optimizer = None

    # ------------------------------------------------------------------ data
    def make_loader(self, episodes: List[Episode], batch_size: Optional[int] = None,
                    human_only: bool = True, num_samples: Optional[int] = None,
                    recency: bool = True) -> Optional[DataLoader]:
        ds = SequenceDataset(
            episodes,
            obs_keys=self.policy.low_dim_keys + self.policy.image_keys,
            image_keys=self.policy.image_keys,
            obs_seq_len=self._seq_lens["obs"],
            action_seq_len=self._seq_lens["action"],
            human_only=human_only,
            mode=self._seq_lens["mode"],
        )
        if len(ds) == 0:
            return None
        bs = int(batch_size or self.tcfg.batch_size)
        bs = max(1, min(bs, len(ds)))
        sampler = None
        shuffle = True
        if recency:
            sampler = ds.recency_sampler(
                self.tcfg.new_data_prob,
                num_samples=num_samples or max(len(ds), bs),
            )
            shuffle = False
        return DataLoader(
            ds, batch_size=bs, sampler=sampler, shuffle=shuffle,
            collate_fn=collate, num_workers=self.tcfg.num_workers,
            drop_last=False, persistent_workers=False,
        )

    def fit_normalizer(self, episodes: List[Episode]) -> None:
        stats = compute_obs_stats(episodes, self.policy.low_dim_keys)
        if stats:
            self.policy.normalizer.fit(stats)
            self.policy.normalizer.to(self.device)

    # ------------------------------------------------------------------ training
    def _step(self, batch: Dict[str, Any], opt: torch.optim.Optimizer
              ) -> Dict[str, float]:
        opt.zero_grad(set_to_none=True)
        loss, metrics = self.policy.compute_loss(batch)
        if not torch.isfinite(loss):
            return {"skipped": 1.0, **metrics}
        loss.backward()
        if self.tcfg.grad_clip and self.tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._params(), self.tcfg.grad_clip)
        opt.step()
        return metrics

    def finetune(
        self,
        episodes: List[Episode],
        steps: Optional[int] = None,
        max_seconds: Optional[float] = None,
        commit: bool = True,
        lr: Optional[float] = None,
        human_only: bool = True,
        progress_cb=None,
    ) -> TrainReport:
        """Run a bounded LoRA update on the aggregated demonstrations.

        Args:
            commit: when False the update is *discarded* after running (sham update,
                used by the non-contingent condition). Wall-clock cost is identical.
            progress_cb: called as `cb(step, total, loss)` for HUD updates.
        """
        report = TrainReport(committed=commit)
        if not self.lora_active:
            self.enable_lora()

        loader = self.make_loader(episodes, human_only=human_only)
        if loader is None:
            report.note = "no trainable samples (no human-labelled steps yet)"
            return report
        report.num_samples = len(loader.dataset)

        total_steps = int(steps if steps is not None else self.tcfg.finetune_steps)
        budget = float(max_seconds if max_seconds is not None
                       else self.tcfg.finetune_max_seconds)
        opt = self._ensure_optimizer(lr)

        # Snapshot before touching anything, so a sham update is exactly reversible.
        #
        # This deliberately captures the *entire* state dict rather than only the
        # trainable tensors. Buffers are not trainable but do change the policy --
        # the observation normaliser's mean/std live in buffers, and an earlier
        # version of this code snapshotted only LoRA parameters, which let a
        # normaliser refit survive a sham update and improve the policy by +0.40
        # success. Anything that alters the policy must be inside the snapshot.
        snapshot = None
        if not commit:
            snapshot = {
                "state": copy.deepcopy(self.policy.state_dict()),
                "optimizer": copy.deepcopy(opt.state_dict()),
            }

        self.policy.train()
        losses: List[float] = []
        agg: Dict[str, List[float]] = {}
        t0 = time.time()
        done = 0
        it = iter(loader)
        while done < total_steps:
            if budget > 0 and (time.time() - t0) > budget:
                report.note = f"time budget {budget:.0f}s reached at step {done}"
                break
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            metrics = self._step(batch, opt)
            done += 1
            key = "nll" if "nll" in metrics else ("loss" if "loss" in metrics else None)
            if key:
                losses.append(float(metrics[key]))
            for k, v in metrics.items():
                agg.setdefault(k, []).append(float(v))
            if progress_cb is not None and (done % 10 == 0 or done == total_steps):
                progress_cb(done, total_steps, losses[-1] if losses else float("nan"))

        self.policy.eval()
        report.steps = done
        report.seconds = time.time() - t0
        if losses:
            head = max(1, len(losses) // 10)
            report.loss_start = float(np.mean(losses[:head]))
            report.loss_end = float(np.mean(losses[-head:]))
        report.metrics = {k: float(np.mean(v[-max(1, len(v) // 10):]))
                          for k, v in agg.items()}

        if snapshot is not None:
            # Roll the policy back: the participant waited, but nothing was learned.
            self.policy.load_state_dict(snapshot["state"], strict=True)
            opt.load_state_dict(snapshot["optimizer"])
            report.note = (report.note + " | sham update rolled back").strip(" |")
        return report

    # ------------------------------------------------------------------ pretraining
    def pretrain(
        self,
        episodes: List[Episode],
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        human_only: bool = True,
        log_every: int = 1,
        logger=print,
    ) -> TrainReport:
        """Full (non-LoRA) training of the base policy from a seed dataset."""
        if self.lora_active:
            raise RuntimeError(
                "pretrain() must run before enable_lora(); the base weights are "
                "frozen once adapters are attached."
            )
        self.fit_normalizer(episodes)
        loader = self.make_loader(episodes, human_only=human_only, recency=False)
        if loader is None:
            raise ValueError("no trainable samples in the seed dataset")

        n_epochs = int(epochs or self.tcfg.pretrain_epochs)
        n_steps = int(steps_per_epoch or self.tcfg.pretrain_steps_per_epoch)
        for p in self.policy.parameters():
            p.requires_grad_(True)
        self.optimizer = None
        opt = self._ensure_optimizer(self.tcfg.base_lr)

        report = TrainReport(num_samples=len(loader.dataset))
        t0 = time.time()
        self.policy.train()
        first_epoch_loss = None
        last_loss = float("nan")
        total = 0
        for ep_i in range(n_epochs):
            it = iter(loader)
            vals: List[float] = []
            for _ in range(n_steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                m = self._step(batch, opt)
                total += 1
                key = "nll" if "nll" in m else ("loss" if "loss" in m else None)
                if key:
                    vals.append(float(m[key]))
            last_loss = float(np.mean(vals)) if vals else float("nan")
            if first_epoch_loss is None:
                first_epoch_loss = last_loss
            if logger and log_every and (ep_i % log_every == 0 or ep_i == n_epochs - 1):
                logger(f"  epoch {ep_i + 1:3d}/{n_epochs}  loss {last_loss:.4f}"
                       f"  ({time.time() - t0:.0f}s)")
        self.policy.eval()
        report.steps = total
        report.seconds = time.time() - t0
        report.loss_start = first_epoch_loss if first_epoch_loss is not None else float("nan")
        report.loss_end = last_loss
        return report

    # ------------------------------------------------------------------ checkpoints
    def save_full(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.policy.save(path, extra=extra)

    def save_adapter(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Round-by-round delta snapshot: a few hundred KB each."""
        torch.save({"lora": lora_state_dict(self.policy), "extra": extra or {}}, path)

    def load_adapter(self, path: str) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(blob["lora"], strict=False)
        self.policy.to(self.device)

    def param_summary(self) -> str:
        tr, tot = count_parameters(self.policy)
        return f"{tr:,} trainable / {tot:,} total"
