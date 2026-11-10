"""The same lifecycle as train.py around a real PyTorch model: a small ResNet
on CIFAR-10. Needs `torch` and `torchvision`, and is not run in CI.

    forge submit --gpus 1 --image <your image> -- python examples/train_torch.py --epochs 5
"""

from __future__ import annotations

import argparse
import io

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import forge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data", default="./data")
    args = ap.parse_args()

    run = forge.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.resnet18(num_classes=10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Resume: model, optimizer and position in the data all come from the checkpoint.
    step, blob = run.restore()
    if blob:
        state = torch.load(io.BytesIO(blob), map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        print(f"resumed from step {step}", flush=True)

    def save(at: int) -> None:
        buf = io.BytesIO()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, buf)
        run.checkpoint(at, buf.getvalue())  # exits here if the job is being preempted

    data = torchvision.datasets.CIFAR10(
        args.data, train=True, download=True,
        transform=T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]),
    )  # fmt: skip
    steps_per_epoch = len(data) // args.batch_size

    for epoch in range(step // steps_per_epoch, args.epochs):
        # Seed the shuffle by epoch so a resumed run sees the same order and can skip ahead.
        gen = torch.Generator().manual_seed(epoch)
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=True,
            generator=gen,
            drop_last=True,
            num_workers=2,
        )
        model.train()
        for i, (x, y) in enumerate(loader):
            at = epoch * steps_per_epoch + i
            if at < step:
                continue
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run.log(at, loss=loss.item(), accuracy=(logits.argmax(1) == y).float().mean().item())
            if run.should_checkpoint():
                save(at + 1)
        step = (epoch + 1) * steps_per_epoch

    save(args.epochs * steps_per_epoch)


if __name__ == "__main__":
    main()
