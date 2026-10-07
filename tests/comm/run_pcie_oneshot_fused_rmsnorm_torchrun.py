"""torchrun driver for the picked #341 PCIe oneshot fused RMSNorm GPU phases.

Runs every phase of ``tests/comm/test_pcie_oneshot_fused_rmsnorm_gpu.py`` with
ALL original assertions intact, under ``torch.distributed.run`` instead of the
pytest ``mp.spawn`` harness. Useful when a hang needs per-phase progress logs
and a guaranteed stack dump: this driver arms
``faulthandler.dump_traceback_later(..., exit=True)`` so any livelock or
deadlock dumps all thread stacks and exits nonzero instead of spinning.

Run from the repository root (8 GPUs visible):

    python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 \
        --master_addr=127.0.0.1 --master_port=<port> \
        tests/comm/run_pcie_oneshot_fused_rmsnorm_torchrun.py

Environment: ``B12X_DRIVER_FAULTHANDLER_SECONDS`` (default 900) bounds the run.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import time
from datetime import timedelta

# Any hang dumps all stacks and exits nonzero (py-spy/gdb are unavailable in
# the qualification image; this is the only reliable stack probe).
faulthandler.dump_traceback_later(
    float(os.environ.get("B12X_DRIVER_FAULTHANDLER_SECONDS", "900")), exit=True
)

import torch
import torch.distributed as dist

sys.path.insert(0, os.getcwd())  # repository root: `tests.*` must import

import tests.comm.test_pcie_oneshot_fused_rmsnorm_gpu as T

PHASES = (
    ("_run_eager", T._run_eager),
    ("_run_graph", T._run_graph),
    ("_run_tp8_graph_mode_transition", T._run_tp8_graph_mode_transition),
    ("_run_tp8_split_view_graph", T._run_tp8_split_view_graph),
    ("_run_pdl_dependent", T._run_pdl_dependent),
)


def log(msg: str) -> None:
    print(f"[rank {os.environ.get('RANK', '?')}] {msg}", flush=True)


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    log("cuda init ok")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=600),
    )
    log("nccl group ok")

    from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool

    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=192 * 1024,
        max_size=192 * 1024,
        max_concurrent_channels=4,
    )
    try:
        t0 = time.time()
        pool.prepare_channels(
            (
                "eager:fused-rmsnorm",
                "graph:fused-rmsnorm",
                "graph:fused-transition",
                "graph:split-residual",
                "graph:pdl-plain-attr",
                "graph:pdl-plain-noattr",
                "graph:pdl-fused-attr",
                "graph:pdl-fused-noattr",
            )
        )
        log(f"prepare_channels(8) ok in {time.time() - t0:.1f}s")
        pool.for_stream(channel_id="eager:fused-rmsnorm")
        log("for_stream ok")

        for name, fn in PHASES:
            dist.barrier()
            t0 = time.time()
            log(f"PHASE START {name}")
            fn(pool, device, rank)
            log(f"PHASE PASS  {name} in {time.time() - t0:.1f}s")
        torch.cuda.synchronize(device)
        dist.barrier()
        log("ALL PHASES PASS")
    finally:
        pool.close()
        dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
