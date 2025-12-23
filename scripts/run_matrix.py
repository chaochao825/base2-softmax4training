import os
import subprocess


BASE = "/home/spco/base-2-bitnet"


def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    common = ["python", f"{BASE}/scripts/train.py", "--dataset", "CIFAR-10", "--epochs", "1", "--batch_size", "256", "--output_dir", f"{BASE}/results"]

    jobs = [
        common + ["--model", "resnet18", "--softmax", "standard"],
        common + ["--model", "resnet18", "--softmax", "base2"],
        common + ["--model", "vit-s", "--softmax", "standard"],
        common + ["--model", "vit-s", "--softmax", "base2"],
    ]

    for j in jobs:
        run(j)


if __name__ == "__main__":
    main()


