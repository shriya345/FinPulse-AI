#!/bin/bash
# Project-local CPU build and quantized model. No package manager or global writes.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p .runtime
cd .runtime
cmake_bin="${CMAKE_BIN:-$(command -v cmake || true)}"
if [[ -z "$cmake_bin" && -x /opt/anaconda3/envs/ctf/bin/cmake ]]; then cmake_bin=/opt/anaconda3/envs/ctf/bin/cmake; fi
[[ -x "$cmake_bin" ]] || { echo "Set CMAKE_BIN to an installed CMake executable."; exit 1; }
fetch() {
  local file="$1" sha="$2" url="$3"
  if [[ -f "$file" ]] && echo "$sha  $file" | shasum -a 256 -c - >/dev/null 2>&1; then return; fi
  curl --fail --location --retry 2 "$url" --output "$file.part"
  echo "$sha  $file.part" | shasum -a 256 -c -
  mv "$file.part" "$file"
}
fetch qwen.gguf 1adf0b11065d8ad2e8123ea110d1ec956dab4ab038eab665614adba04b6c3370 'https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf?download=true'
fetch llama.tar.gz 2abe4bb29e4f4815b2b1ce905f425063f378afd570c9547cd092cc1f92b932af 'https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/b5430'
tar -xzf llama.tar.gz
"$cmake_bin" -S llama.cpp-b5430 -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
"$cmake_bin" --build build --target llama-server -j 6
