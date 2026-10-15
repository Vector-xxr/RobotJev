#!/usr/bin/env python3
"""GPU 不可用时，在 CPU 上起 PlayJev 服务。

    python serve_cpu.py --ckpt weights/PlayJev-0.8B --port 18732

**为什么不能直接 `playjev.serve --device cpu`。** Qwen3.5 的线性注意力层在
transformers 里是这么选实现的（`integrations/hub_kernels.py`
`use_kernel_func_from_hub_with_fallback`）：**导入时** `import fla` 成功就用
flash-linear-attention 的 Triton 内核，失败才退回纯 PyTorch。Triton 只能跑在 GPU 上，
所以装着 fla 却没有 GPU 时，每个请求都会 500：

    RuntimeError: 0 active drivers ([]). There should only be one.

这里在进程里把 `fla` 挡掉，让 transformers 走纯 PyTorch 路径。不卸 fla ——
GPU 修好之后照常用 `python -m playjev.serve`，它还要那个快的内核。
"""

import sys

# 用 sys.modules[...] = None 挡，不用自定义 finder 抛异常：playjev/model.py 里有一句
# `importlib.util.find_spec("fla")` 探测内核可用性，finder 抛异常会把这句探测也炸掉，
# 服务直接起不来。置 None 之后 find_spec 老实返回 None，`import fla` 抛 ImportError，
# transformers 的回退逻辑（它 catch 的是 Exception）照常退到纯 PyTorch。
for _name in ("fla", "fla.ops", "fla.modules"):
    sys.modules[_name] = None

if "--device" not in sys.argv:
    sys.argv += ["--device", "cpu"]

from playjev.serve import main  # noqa: E402

if __name__ == "__main__":
    main()
