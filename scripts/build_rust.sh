#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cargo build --manifest-path rust/Cargo.toml --release
cp rust/target/release/libsgad_rust_native.so src/sgad/rust/sgad_rust_native.so

echo "Built and copied src/sgad/rust/sgad_rust_native.so"
