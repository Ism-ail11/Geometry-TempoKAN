"""Build or load the optional, first-order CPU spline kernel."""

import argparse
import ctypes
import os
from pathlib import Path
import subprocess

import torch


def load_kernel():
    filename = os.environ.get("TEMPO_KAN_NATIVE")
    if not filename:
        return None
    library = ctypes.CDLL(str(Path(filename).resolve()))
    pointer = ctypes.c_void_p
    dimensions = [ctypes.c_int64, ctypes.c_int, ctypes.c_int]
    library.tempo_cubic_forward.argtypes = [pointer, pointer, *dimensions]
    library.tempo_cubic_backward.argtypes = [pointer, pointer, pointer, *dimensions]
    library.tempo_cubic_forward.restype = None
    library.tempo_cubic_backward.restype = None
    return library


KERNEL = load_kernel()


class CPUTempoCubic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grid, with_base):
        samples = x.contiguous()
        output = torch.empty((*samples.shape, grid + 3 + with_base), dtype=samples.dtype, device=samples.device)
        KERNEL.tempo_cubic_forward(samples.data_ptr(), output.data_ptr(), samples.numel(), grid, with_base)
        ctx.save_for_backward(samples)
        ctx.grid, ctx.with_base = grid, with_base
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        samples, = ctx.saved_tensors
        grad = gradient.contiguous()
        output = torch.empty_like(samples)
        KERNEL.tempo_cubic_backward(samples.data_ptr(), grad.data_ptr(), output.data_ptr(), samples.numel(),
                                   ctx.grid, ctx.with_base)
        return output, None, None


def build(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("cubic_cpu.cpp")
    subprocess.run([os.environ.get("CXX", "c++"), "-O3", "-DNDEBUG", "-std=c++17", "-ffp-contract=off",
                    "-shared", "-fPIC", str(source), "-o", str(output)], check=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/tempo_cubic.so"))
    print(build(parser.parse_args().output))


if __name__ == "__main__":
    main()
