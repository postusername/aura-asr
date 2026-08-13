#!/bin/bash
# Сборка whisper.cpp из исходников. Тег зафиксирован здесь, а не выбирается
# на лету — воспроизводимость сборки важнее свежести.
set -euo pipefail

WHISPER_TAG=v1.9.2
VENDOR_DIR="$(dirname "$0")/../vendor/whisper.cpp"

if [ ! -d "$VENDOR_DIR" ]; then
    git clone --depth 1 --branch "$WHISPER_TAG" \
        https://github.com/ggml-org/whisper.cpp.git "$VENDOR_DIR"
fi

cmake -B "$VENDOR_DIR/build" -S "$VENDOR_DIR" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DWHISPER_BUILD_SERVER=ON
cmake --build "$VENDOR_DIR/build" --config Release -j"$(nproc)"

install -m 755 "$VENDOR_DIR"/build/bin/whisper-server /usr/local/bin/
install -m 755 "$VENDOR_DIR"/build/bin/whisper-cli /usr/local/bin/
install -m 755 "$VENDOR_DIR"/build/bin/whisper-quantize /usr/local/bin/
cp "$VENDOR_DIR"/build/bin/*.so* /usr/local/lib/
ldconfig
