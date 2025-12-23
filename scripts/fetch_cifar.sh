#!/usr/bin/env bash
set -euo pipefail

# Download CIFAR archives from mirror into the expected filenames so torchvision can reuse them offline.
# Usage: bash scripts/fetch_cifar.sh [DATA_DIR]

DATA_DIR=${1:-/home/spco/data}
mkdir -p "${DATA_DIR}"

cd "${DATA_DIR}"

# Helper to robustly download with mirror then fallback
download_with_fallback() {
  local url_mirror="$1"; shift
  local url_official="$1"; shift
  local outfile="$1"; shift

  if [ ! -f "${outfile}" ]; then
    echo "Trying mirror: ${url_mirror}"
    wget -q --show-progress -O "${outfile}.tmp" "${url_mirror}" || true
    # Basic size check (>1MB)
    if [ -f "${outfile}.tmp" ] && [ $(stat -c%s "${outfile}.tmp") -gt 1000000 ]; then
      mv "${outfile}.tmp" "${outfile}"
    else
      echo "Mirror download failed or too small, falling back to official: ${url_official}"
      rm -f "${outfile}.tmp"
      wget -q --show-progress -O "${outfile}" "${url_official}"
    fi
  fi
}

# CIFAR-10
download_with_fallback \
  "https://mirrors.tuna.tsinghua.edu.cn/github-release/krizhevsky/cifar-10-python/cifar-10-python.tar.gz" \
  "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" \
  "cifar-10-python.tar.gz"
echo "Extracting CIFAR-10..."
if ! tar -tzf cifar-10-python.tar.gz >/dev/null 2>&1; then
  echo "CIFAR-10 archive seems corrupted; re-downloading from official..."
  rm -f cifar-10-python.tar.gz
  wget -q --show-progress -O cifar-10-python.tar.gz "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
fi
tar -xzf cifar-10-python.tar.gz

# CIFAR-100
download_with_fallback \
  "https://mirrors.tuna.tsinghua.edu.cn/github-release/krizhevsky/cifar-100-python/cifar-100-python.tar.gz" \
  "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz" \
  "cifar-100-python.tar.gz"
echo "Extracting CIFAR-100..."
if ! tar -tzf cifar-100-python.tar.gz >/dev/null 2>&1; then
  echo "CIFAR-100 archive seems corrupted; re-downloading from official..."
  rm -f cifar-100-python.tar.gz
  wget -q --show-progress -O cifar-100-python.tar.gz "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
fi
tar -xzf cifar-100-python.tar.gz

echo "Done. Archives and extracted folders placed under ${DATA_DIR}."


