#!/usr/bin/env bash
set -euo pipefail

echo "=== Kakip setup script start ==="

# 更新
echo "[1/5] Updating apt repositories..."
sudo apt-get update
sudo apt-get upgrade -y

# 依存インストール
echo "[2/5] Installing required packages..."
sudo apt-get install -y build-essential cmake wget git python3-pip python3-tk

# Python ライブラリ
echo "[3/5] Installing Python libraries..."
pip3 install --upgrade pip
pip3 install paho-mqtt

# kakip_ai_apps リポジトリ取得
WORK=${WORK:-$HOME}
cd "$WORK"
if [ ! -d "kakip_ai_apps" ]; then
    echo "[4/5] Cloning kakip_ai_apps repository..."
    git clone https://github.com/Kakip-ai/kakip_ai_apps --recursive
else
    echo "[4/5] kakip_ai_apps already exists, skipping clone."
fi

# R01_object_detection 向けのヘッダコピー
echo "[5/5] Preparing R01_object_detection headers..."
cd kakip_ai_apps
cp 3rdparty/rzv_drp-ai_tvm/setup/include/* 3rdparty/rzv_drp-ai_tvm/tvm/include/tvm/runtime/ || true

echo "=== Kakip setup script completed ==="
echo "次のステップ:"
echo "1. cd $WORK/kakip_ai_apps/R01_object_detection"
echo "2. README.md の手順に従って動作確認"
echo "3. main.cpp を編集して yolo_to_fifo_writer を作成"
