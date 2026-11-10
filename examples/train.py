"""A small, dependency-free training job that uses every part of the SDK.

It fits logistic regression with minibatch SGD on synthetic data. The model is
trivial on purpose: the point is the lifecycle (metrics, periodic checkpoints,
checkpoint-on-preemption, resume), which is the same for a real network.
See examples/train_torch.py for the same structure around a PyTorch model.

    forge submit --gpus 2 -- python examples/train.py --steps 2000

`--fail` makes the job die in a recognisable way, for exercising triage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import forge

DIM = 16


def make_batch(rng: random.Random, truth: list[float], n: int) -> list[tuple[list[float], int]]:
    batch = []
    for _ in range(n):
        x = [rng.gauss(0, 1) for _ in range(DIM)]
        z = sum(w * xi for w, xi in zip(truth, x, strict=True))
        batch.append((x, 1 if z + rng.gauss(0, 0.5) > 0 else 0))
    return batch


def step(
    weights: list[float], batch: list[tuple[list[float], int]], lr: float
) -> tuple[float, float]:
    grad = [0.0] * DIM
    loss = correct = 0.0
    for x, y in batch:
        z = sum(w * xi for w, xi in zip(weights, x, strict=True))
        p = 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))
        loss -= math.log(p if y else 1 - p + 1e-12)
        correct += (p > 0.5) == bool(y)
        for i, xi in enumerate(x):
            grad[i] += (p - y) * xi
    for i in range(DIM):
        weights[i] -= lr * grad[i] / len(batch)
    return loss / len(batch), correct / len(batch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--sleep", type=float, default=0.02, help="seconds per step, to simulate work")
    ap.add_argument("--checkpoint-seconds", type=float, default=600.0)
    ap.add_argument("--data", default=None, help="optional dataset path (must exist if given)")
    ap.add_argument("--fail", choices=["oom", "nan", "bug"], help="fail on purpose at step 20")
    args = ap.parse_args()

    run = forge.init(checkpoint_seconds=args.checkpoint_seconds)
    if args.data:
        open(args.data).close()

    start, blob = run.restore()
    weights = json.loads(blob)["weights"] if blob else [0.0] * DIM
    if blob:
        print(f"resumed from step {start}", flush=True)

    truth = [random.Random(0).gauss(0, 1) for _ in range(DIM)]
    for i in range(start, args.steps):
        rng = random.Random(i)  # the batch depends only on the step, so a resume is exact
        loss, acc = step(weights, make_batch(rng, truth, args.batch_size), args.lr)

        if args.fail and i == 20:
            if args.fail == "oom":
                raise RuntimeError(
                    f"CUDA out of memory. Tried to allocate {args.batch_size * 48} MiB "
                    "(GPU 0; 23.69 GiB total capacity)"
                )
            if args.fail == "nan":
                run.log(i, loss=float("nan"), accuracy=acc)
                run.flush()
                raise FloatingPointError(f"loss is nan at step {i} (lr={args.lr})")
            raise KeyError("optimizer_state")

        run.log(i, loss=loss, accuracy=acc)
        if i % 50 == 0:
            print(f"step {i} loss {loss:.4f} acc {acc:.3f}", flush=True)
        time.sleep(args.sleep)
        if run.should_checkpoint():
            run.checkpoint(i + 1, json.dumps({"weights": weights}).encode())

    run.checkpoint(args.steps, json.dumps({"weights": weights}).encode())
    print(f"done: loss {loss:.4f} acc {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
