#!/usr/bin/env python3
"""Print PaddlePaddle CUDA runtime details."""

from __future__ import annotations

import paddle


def main() -> None:
    print(f"Paddle version: {paddle.__version__}")
    print(f"Compiled with CUDA: {paddle.device.is_compiled_with_cuda()}")

    if not paddle.device.is_compiled_with_cuda():
        return

    count = paddle.device.cuda.device_count()
    print(f"CUDA device count: {count}")
    for index in range(count):
        try:
            name = paddle.device.cuda.get_device_name(index)
        except Exception:
            name = "unknown"
        print(f"- GPU {index}: {name}")


if __name__ == "__main__":
    main()
