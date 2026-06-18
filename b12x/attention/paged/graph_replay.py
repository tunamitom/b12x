"""Device-side decode graph replay helpers for the paged attention backend."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

_DECODE_BLOCK_CHUNKS = 128
_DECODE_BLOCK_PAGES = 128
_PREFILL_BLOCK_WORK_ITEMS = 128
_PREFILL_BLOCK_ROWS = 128


@triton.jit
def build_decode_graph_page_table_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_table_ptr,
    active_max_pages_ptr,
    req_to_token_row_stride,
    page_table_row_stride,
    PAGE_SIZE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    page_block_idx = tl.program_id(axis=1)

    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx).to(tl.int64)
    active_max_pages = tl.load(active_max_pages_ptr).to(tl.int32)
    page_offsets = page_block_idx * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    page_mask = page_offsets < active_max_pages
    flat_token_offsets = req_pool_idx * req_to_token_row_stride + page_offsets.to(tl.int64) * PAGE_SIZE
    token_indices = tl.load(req_to_token_ptr + flat_token_offsets, mask=page_mask, other=0)
    tl.store(
        page_table_ptr + req_idx * page_table_row_stride + page_offsets,
        (token_indices // PAGE_SIZE).to(tl.int32),
        mask=page_mask,
    )


@triton.jit
def update_decode_graph_metadata_triton(
    cache_seqlens_ptr,
    request_indices_ptr,
    qo_tile_indices_ptr,
    kv_tile_indices_ptr,
    merge_indptr_ptr,
    block_valid_mask_ptr,
    kv_window_start_tokens_ptr,
    chunk_pages_ptr,
    max_chunks_per_req,
    PAGE_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    chunk_block_idx = tl.program_id(axis=1)

    cache_len = tl.load(cache_seqlens_ptr + req_idx).to(tl.int32)
    chunk_pages = tl.load(chunk_pages_ptr).to(tl.int32)
    num_pages = tl.maximum((cache_len + (PAGE_SIZE - 1)) // PAGE_SIZE, 1)
    window_start_page = tl.full((), 0, tl.int32)
    if WINDOW_LEFT >= 0:
        window_start_token = tl.maximum(cache_len - 1 - WINDOW_LEFT, 0)
        window_start_page = window_start_token // PAGE_SIZE
    tl.store(kv_window_start_tokens_ptr + req_idx, window_start_page * PAGE_SIZE)
    effective_pages = num_pages - window_start_page
    num_chunks = (effective_pages + chunk_pages - 1) // chunk_pages

    tl.store(merge_indptr_ptr + req_idx + 1, num_chunks)

    chunk_offsets = chunk_block_idx * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
    chunk_mask = chunk_offsets < max_chunks_per_req
    is_active = chunk_offsets < num_chunks
    work_offsets = req_idx * max_chunks_per_req + chunk_offsets
    tl.store(
        request_indices_ptr + work_offsets,
        req_idx,
        mask=chunk_mask,
    )
    tl.store(
        qo_tile_indices_ptr + work_offsets,
        0,
        mask=chunk_mask,
    )
    tl.store(
        kv_tile_indices_ptr + work_offsets,
        chunk_offsets.to(tl.int32),
        mask=chunk_mask,
    )
    tl.store(
        block_valid_mask_ptr + work_offsets,
        is_active.to(tl.int32),
        mask=chunk_mask,
    )


@triton.jit
def update_regular_decode_graph_metadata_triton(
    cache_seqlens_ptr,
    merge_indptr_ptr,
    kv_window_start_tokens_ptr,
    chunk_pages_ptr,
    PAGE_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)

    cache_len = tl.load(cache_seqlens_ptr + req_idx).to(tl.int32)
    chunk_pages = tl.load(chunk_pages_ptr).to(tl.int32)
    kv_chunk_size = chunk_pages * PAGE_SIZE
    num_pages = tl.maximum((cache_len + (PAGE_SIZE - 1)) // PAGE_SIZE, 1)
    window_start_page = tl.full((), 0, tl.int32)
    if WINDOW_LEFT >= 0:
        window_start_token = tl.maximum(cache_len - 1 - WINDOW_LEFT, 0)
        window_start_page = window_start_token // PAGE_SIZE
    tl.store(kv_window_start_tokens_ptr + req_idx, window_start_page * PAGE_SIZE)
    effective_pages = num_pages - window_start_page
    num_chunks = tl.maximum((effective_pages * PAGE_SIZE + kv_chunk_size - 1) // kv_chunk_size, 1)

    tl.store(merge_indptr_ptr + req_idx + 1, num_chunks)


@triton.jit
def update_prefill_graph_work_metadata_triton(
    cache_seqlens_ptr,
    cu_seqlens_q_ptr,
    request_indices_ptr,
    qo_tile_indices_ptr,
    kv_tile_indices_ptr,
    o_indptr_ptr,
    block_valid_mask_ptr,
    kv_chunk_size_ptr,
    kv_window_start_tokens_ptr,
    work_items_capacity: tl.constexpr,
    block_valid_capacity: tl.constexpr,
    BATCH: tl.constexpr,
    MAX_Q_TILES_PER_REQ: tl.constexpr,
    MAX_CHUNKS_PER_Q_TILE: tl.constexpr,
    CTA_TILE_Q: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    BLOCK_WORK_ITEMS: tl.constexpr,
):
    block_idx = tl.program_id(axis=0)
    offsets = block_idx * BLOCK_WORK_ITEMS + tl.arange(0, BLOCK_WORK_ITEMS)
    slots_per_req = MAX_Q_TILES_PER_REQ * MAX_CHUNKS_PER_Q_TILE
    req_idx = offsets // slots_per_req
    req_local = offsets - req_idx * slots_per_req

    in_block_capacity = offsets < block_valid_capacity
    in_work_capacity = offsets < work_items_capacity
    in_batch = req_idx < BATCH
    usable = in_block_capacity & in_work_capacity & in_batch

    q_start = tl.load(cu_seqlens_q_ptr + req_idx, mask=usable, other=0).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + req_idx + 1, mask=usable, other=0).to(
        tl.int32
    )
    q_len = tl.maximum(q_end - q_start, 0)
    cache_len = tl.load(cache_seqlens_ptr + req_idx, mask=usable, other=0).to(tl.int32)

    packed_q_len = q_len * GQA_GROUP_SIZE
    num_q_tiles = (packed_q_len + CTA_TILE_Q - 1) // CTA_TILE_Q
    num_pages = tl.maximum((cache_len + PAGE_SIZE - 1) // PAGE_SIZE, 1)

    window_start_page = tl.full((BLOCK_WORK_ITEMS,), 0, tl.int32)
    if WINDOW_LEFT >= 0:
        first_causal_key = tl.maximum(cache_len - tl.maximum(q_len, 1), 0)
        first_window_key = tl.maximum(first_causal_key - WINDOW_LEFT, 0)
        window_start_page = first_window_key // PAGE_SIZE
        window_start_page = tl.minimum(window_start_page, tl.maximum(num_pages - 1, 0))
    effective_pages = tl.maximum(num_pages - window_start_page, 1)

    kv_chunk_size = tl.load(kv_chunk_size_ptr).to(tl.int32)
    chunk_pages = tl.maximum((kv_chunk_size + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    num_chunks = tl.full((BLOCK_WORK_ITEMS,), 1, tl.int32)
    if SPLIT_KV:
        num_chunks = tl.maximum((effective_pages + chunk_pages - 1) // chunk_pages, 1)
    # Safety: the work-item grid only has MAX_CHUNKS_PER_Q_TILE slots per q tile.
    # If the runtime chunk count ever exceeds the primed capacity, clamp so that
    # o_indptr/merge_indptr stay consistent with the work items actually
    # scheduled (otherwise the merge consumes uninitialized partial rows).
    num_chunks = tl.minimum(num_chunks, MAX_CHUNKS_PER_Q_TILE)

    # The forward kernel stores split-KV partials densely as
    #   token_local * num_chunks + kv_tile_idx
    # for each request.  CUDA graph replay still reserves the maximum captured
    # slots per request, but the active runtime work items must be packed by the
    # runtime chunk count instead of leaving MAX_CHUNKS_PER_Q_TILE-sized gaps.
    safe_num_chunks = tl.maximum(num_chunks, 1)
    q_tile_idx = req_local // safe_num_chunks
    kv_tile_idx = req_local - q_tile_idx * safe_num_chunks

    active = (
        usable
        & (q_tile_idx < num_q_tiles)
        & (kv_tile_idx < num_chunks)
        & (q_len > 0)
        & (cache_len > 0)
    )
    tl.store(block_valid_mask_ptr + offsets, active.to(tl.int32), mask=in_block_capacity)
    tl.store(request_indices_ptr + offsets, req_idx.to(tl.int32), mask=in_work_capacity)
    tl.store(qo_tile_indices_ptr + offsets, q_tile_idx.to(tl.int32), mask=in_work_capacity)
    tl.store(kv_tile_indices_ptr + offsets, kv_tile_idx.to(tl.int32), mask=in_work_capacity)

    first_slot_for_req = usable & (req_local == 0)
    tl.store(
        kv_window_start_tokens_ptr + req_idx,
        (window_start_page * PAGE_SIZE).to(tl.int32),
        mask=first_slot_for_req,
    )
    tl.store(
        o_indptr_ptr + req_idx + 1,
        (q_len * num_chunks).to(tl.int32),
        mask=first_slot_for_req,
    )
    tl.store(o_indptr_ptr + offsets, 0, mask=offsets == 0)


@triton.jit
def update_prefill_graph_row_indptr_triton(
    cu_seqlens_q_ptr,
    o_indptr_ptr,
    merge_indptr_ptr,
    BATCH: tl.constexpr,
    MAX_Q_ROWS_PER_REQ: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    row_block_idx = tl.program_id(axis=1)
    rows = row_block_idx * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)

    q_start = tl.load(cu_seqlens_q_ptr + req_idx).to(tl.int32)
    q_end = tl.load(cu_seqlens_q_ptr + req_idx + 1).to(tl.int32)
    q_len = tl.maximum(q_end - q_start, 0)
    request_partial_start = tl.load(o_indptr_ptr + req_idx).to(tl.int32)
    request_partial_end = tl.load(o_indptr_ptr + req_idx + 1).to(tl.int32)
    safe_q_len = tl.maximum(q_len, 1)
    num_chunks = tl.maximum(
        (request_partial_end - request_partial_start) // safe_q_len, 1
    )

    row_mask = rows < q_len
    tl.store(
        merge_indptr_ptr + q_start + rows + 1,
        request_partial_start + (rows + 1) * num_chunks,
        mask=row_mask,
    )
    if BATCH > 0:
        tl.store(merge_indptr_ptr, 0, mask=(req_idx == 0) & (row_block_idx == 0))


def make_decode_chunk_pages_lut_tensor(
    decode_chunk_pages_lut: Sequence[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not decode_chunk_pages_lut:
        raise ValueError("decode chunk-pages LUT must be non-empty")
    if any(int(chunk_pages) <= 0 for chunk_pages in decode_chunk_pages_lut):
        raise ValueError("decode chunk-pages LUT must contain only positive values")
    return torch.tensor(
        (int(decode_chunk_pages_lut[0]), *(int(chunk_pages) for chunk_pages in decode_chunk_pages_lut)),
        dtype=torch.int32,
        device=device,
    )


def summarize_decode_chunk_pages_lut(
    decode_chunk_pages_lut: Sequence[int],
) -> tuple[int, int]:
    if not decode_chunk_pages_lut:
        raise ValueError("decode chunk-pages LUT must be non-empty")
    worst_page_count = 1
    max_chunks_per_req = 1
    for page_count, chunk_pages in enumerate(decode_chunk_pages_lut, start=1):
        num_chunks = (page_count + int(chunk_pages) - 1) // int(chunk_pages)
        if num_chunks > max_chunks_per_req:
            max_chunks_per_req = num_chunks
            worst_page_count = page_count
    return int(worst_page_count), int(max_chunks_per_req)


def update_decode_graph_replay_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
    window_left: int = -1,
) -> None:
    if req_to_token.device != page_table.device:
        raise ValueError("req_to_token and page_table must be on the same device")
    if req_pool_indices.device != page_table.device:
        raise ValueError("req_pool_indices and page_table must be on the same device")
    if cache_seqlens.device != page_table.device:
        raise ValueError("cache_seqlens and page_table must be on the same device")
    if qo_tile_indices.device != page_table.device or kv_tile_indices.device != page_table.device:
        raise ValueError("tile index buffers and page_table must be on the same device")
    if decode_chunk_pages_lut.device != page_table.device:
        raise ValueError("decode_chunk_pages_lut and page_table must be on the same device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    if int(req_pool_indices.shape[0]) != bs:
        raise ValueError("req_pool_indices shape must match cache_seqlens batch")
    work_items_capacity = int(request_indices.shape[0])
    if int(qo_tile_indices.shape[0]) != work_items_capacity or int(kv_tile_indices.shape[0]) != work_items_capacity:
        raise RuntimeError("decode graph tile index buffers must match request_indices shape")
    if work_items_capacity % bs != 0:
        raise RuntimeError("decode graph workspace request_indices shape is incompatible with the batch bucket")
    max_chunks_per_req = work_items_capacity // bs
    if max_chunks_per_req <= 0:
        raise RuntimeError("decode graph workspace must allocate at least one chunk per request")

    max_cache_pages = torch.div(
        cache_seqlens[:bs].amax() + (page_size - 1),
        page_size,
        rounding_mode="floor",
    ).clamp_(min=1, max=page_table.shape[1]).to(torch.int64)
    active_max_pages = max_cache_pages.to(torch.int32)
    if window_page_span > 0:
        effective_max_pages = max_cache_pages.clamp(max=int(window_page_span))
    else:
        effective_max_pages = max_cache_pages
    effective_max_pages = effective_max_pages.clamp_(min=1, max=decode_chunk_pages_lut.shape[0] - 1)
    decode_chunk_pages = torch.index_select(decode_chunk_pages_lut, 0, effective_max_pages.view(1))

    page_blocks = triton.cdiv(int(page_table.shape[1]), _DECODE_BLOCK_PAGES)
    build_decode_graph_page_table_triton[(bs, page_blocks)](
        req_to_token,
        req_pool_indices,
        page_table,
        active_max_pages,
        req_to_token.stride(0),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )

    block_valid_mask.zero_()
    merge_indptr.zero_()
    chunk_blocks = triton.cdiv(max_chunks_per_req, _DECODE_BLOCK_CHUNKS)
    update_decode_graph_metadata_triton[(bs, chunk_blocks)](
        cache_seqlens,
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        merge_indptr,
        block_valid_mask,
        kv_window_start_tokens,
        decode_chunk_pages,
        max_chunks_per_req,
        PAGE_SIZE=page_size,
        WINDOW_LEFT=int(window_left),
        BLOCK_CHUNKS=_DECODE_BLOCK_CHUNKS,
    )
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    kv_chunk_size_ptr.copy_(decode_chunk_pages * page_size)


def update_regular_decode_graph_replay_metadata(
    *,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
    window_left: int = -1,
) -> None:
    if req_to_token.device != page_table.device:
        raise ValueError("req_to_token and page_table must be on the same device")
    if req_pool_indices.device != page_table.device:
        raise ValueError("req_pool_indices and page_table must be on the same device")
    if cache_seqlens.device != page_table.device:
        raise ValueError("cache_seqlens and page_table must be on the same device")
    if decode_chunk_pages_lut.device != page_table.device:
        raise ValueError("decode_chunk_pages_lut and page_table must be on the same device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    if int(req_pool_indices.shape[0]) != bs:
        raise ValueError("req_pool_indices shape must match cache_seqlens batch")

    max_cache_pages = torch.div(
        cache_seqlens[:bs].amax() + (page_size - 1),
        page_size,
        rounding_mode="floor",
    ).clamp_(min=1, max=page_table.shape[1]).to(torch.int64)
    active_max_pages = max_cache_pages.to(torch.int32)
    if window_page_span > 0:
        effective_max_pages = max_cache_pages.clamp(max=int(window_page_span))
    else:
        effective_max_pages = max_cache_pages
    effective_max_pages = effective_max_pages.clamp_(min=1, max=decode_chunk_pages_lut.shape[0] - 1)
    decode_chunk_pages = torch.index_select(decode_chunk_pages_lut, 0, effective_max_pages.view(1))

    page_blocks = triton.cdiv(int(page_table.shape[1]), _DECODE_BLOCK_PAGES)
    build_decode_graph_page_table_triton[(bs, page_blocks)](
        req_to_token,
        req_pool_indices,
        page_table,
        active_max_pages,
        req_to_token.stride(0),
        page_table.stride(0),
        PAGE_SIZE=page_size,
        BLOCK_PAGES=_DECODE_BLOCK_PAGES,
    )

    merge_indptr.zero_()
    update_regular_decode_graph_metadata_triton[(bs,)](
        cache_seqlens,
        merge_indptr,
        kv_window_start_tokens,
        decode_chunk_pages,
        PAGE_SIZE=page_size,
        WINDOW_LEFT=int(window_left),
    )
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    kv_chunk_size_ptr.copy_(decode_chunk_pages * page_size)


def update_regular_decode_graph_chunk_metadata(
    *,
    cache_seqlens: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_chunk_size: int | torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    max_chunks_per_req: int,
    page_size: int,
    window_page_span: int = 0,
    window_left: int = -1,
) -> None:
    device = cache_seqlens.device
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if kv_chunk_size_ptr.device != device:
        raise ValueError("decode graph buffers and cache_seqlens must be on the same device")
    if kv_window_start_tokens.device != device:
        raise ValueError("kv_window_start_tokens and cache_seqlens must be on the same device")
    if max_chunks_per_req <= 0:
        raise ValueError("max_chunks_per_req must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")

    if isinstance(kv_chunk_size, torch.Tensor):
        if kv_chunk_size.device != device:
            raise ValueError("kv_chunk_size tensor and cache_seqlens must be on the same device")
        if kv_chunk_size.numel() != 1:
            raise ValueError("kv_chunk_size tensor must contain exactly one element")
        kv_chunk_size_i32 = kv_chunk_size.reshape(1).to(torch.int32)
    else:
        if kv_chunk_size <= 0:
            raise ValueError("kv_chunk_size must be positive")
        kv_chunk_size_i32 = None

    merge_indptr.zero_()
    num_pages = torch.maximum(
        torch.div(
            cache_seqlens[:bs] + (int(page_size) - 1),
            int(page_size),
            rounding_mode="floor",
        ),
        torch.ones(bs, dtype=torch.int32, device=device),
    )
    if window_left >= 0:
        window_start_tokens = (cache_seqlens[:bs] - 1 - int(window_left)).clamp(min=0)
        window_start_pages = torch.div(
            window_start_tokens,
            int(page_size),
            rounding_mode="floor",
        ).to(torch.int32)
    else:
        window_start_pages = torch.zeros(bs, dtype=torch.int32, device=device)
    effective_pages = num_pages - window_start_pages
    kv_window_start_tokens[:bs].copy_(window_start_pages * int(page_size))
    effective_tokens = effective_pages * int(page_size)
    if kv_chunk_size_i32 is None:
        num_chunks = torch.maximum(
            torch.div(
                effective_tokens + (int(kv_chunk_size) - 1),
                int(kv_chunk_size),
                rounding_mode="floor",
            ),
            torch.ones(bs, dtype=torch.int32, device=device),
        )
    else:
        num_chunks = torch.maximum(
            torch.div(
                effective_tokens + (kv_chunk_size_i32 - 1),
                kv_chunk_size_i32,
                rounding_mode="floor",
            ),
            torch.ones(bs, dtype=torch.int32, device=device),
        )
    merge_indptr[1 : bs + 1].copy_(num_chunks)
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    if kv_chunk_size_i32 is None:
        kv_chunk_size_ptr[0] = int(kv_chunk_size)
    else:
        kv_chunk_size_ptr[:1].copy_(kv_chunk_size_i32.to(kv_chunk_size_ptr.dtype))


def update_decode_graph_chunk_metadata(
    *,
    cache_seqlens: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    decode_chunk_pages_lut: torch.Tensor,
    page_size: int,
    window_page_span: int = 0,
    window_left: int = -1,
) -> None:
    device = cache_seqlens.device
    if request_indices.device != device:
        raise ValueError("request_indices and cache_seqlens must be on the same device")
    if qo_tile_indices.device != device or kv_tile_indices.device != device:
        raise ValueError("tile index buffers and cache_seqlens must be on the same device")
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if block_valid_mask.device != device or kv_chunk_size_ptr.device != device:
        raise ValueError("decode graph buffers and cache_seqlens must be on the same device")
    if decode_chunk_pages_lut.device != device:
        raise ValueError("decode_chunk_pages_lut and cache_seqlens must be on the same device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    bs = int(cache_seqlens.shape[0])
    if bs <= 0:
        raise ValueError("decode graph replay requires bs > 0")
    work_items_capacity = int(request_indices.shape[0])
    if int(qo_tile_indices.shape[0]) != work_items_capacity or int(kv_tile_indices.shape[0]) != work_items_capacity:
        raise RuntimeError("decode graph tile index buffers must match request_indices shape")
    if work_items_capacity % bs != 0:
        raise RuntimeError("decode graph workspace request_indices shape is incompatible with the batch bucket")
    max_chunks_per_req = work_items_capacity // bs
    if max_chunks_per_req <= 0:
        raise RuntimeError("decode graph workspace must allocate at least one chunk per request")

    max_cache_pages = torch.div(
        cache_seqlens[:bs].amax() + (page_size - 1),
        page_size,
        rounding_mode="floor",
    ).clamp_(min=1, max=decode_chunk_pages_lut.shape[0] - 1).to(torch.int64)
    if window_page_span > 0:
        effective_max_pages = max_cache_pages.clamp(max=int(window_page_span))
    else:
        effective_max_pages = max_cache_pages
    effective_max_pages = effective_max_pages.clamp_(min=1, max=decode_chunk_pages_lut.shape[0] - 1)
    decode_chunk_pages = torch.index_select(decode_chunk_pages_lut, 0, effective_max_pages.view(1))

    block_valid_mask.zero_()
    merge_indptr.zero_()
    chunk_blocks = triton.cdiv(max_chunks_per_req, _DECODE_BLOCK_CHUNKS)
    update_decode_graph_metadata_triton[(bs, chunk_blocks)](
        cache_seqlens,
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        merge_indptr,
        block_valid_mask,
        kv_window_start_tokens,
        decode_chunk_pages,
        max_chunks_per_req,
        PAGE_SIZE=page_size,
        WINDOW_LEFT=int(window_left),
        BLOCK_CHUNKS=_DECODE_BLOCK_CHUNKS,
    )
    torch.cumsum(
        merge_indptr[1 : bs + 1],
        dim=0,
        out=merge_indptr[1 : bs + 1],
    )
    o_indptr[: bs + 1].copy_(merge_indptr[: bs + 1])
    kv_chunk_size_ptr.copy_(decode_chunk_pages * page_size)


def update_prefill_graph_chunk_metadata(
    *,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    request_indices: torch.Tensor,
    qo_tile_indices: torch.Tensor,
    kv_tile_indices: torch.Tensor,
    merge_indptr: torch.Tensor,
    o_indptr: torch.Tensor,
    block_valid_mask: torch.Tensor,
    kv_chunk_size_ptr: torch.Tensor,
    kv_window_start_tokens: torch.Tensor,
    total_num_rows_ptr: torch.Tensor,
    batch: int,
    max_q_tiles_per_req: int,
    max_chunks_per_q_tile: int,
    max_q_rows_per_req: int,
    cta_tile_q: int,
    gqa_group_size: int,
    page_size: int,
    split_kv: bool,
    window_left: int = -1,
) -> None:
    device = cache_seqlens.device
    if cu_seqlens_q.device != device:
        raise ValueError("cu_seqlens_q and cache_seqlens must be on the same device")
    if request_indices.device != device:
        raise ValueError("request_indices and cache_seqlens must be on the same device")
    if qo_tile_indices.device != device or kv_tile_indices.device != device:
        raise ValueError("tile index buffers and cache_seqlens must be on the same device")
    if merge_indptr.device != device or o_indptr.device != device:
        raise ValueError("indptr buffers and cache_seqlens must be on the same device")
    if block_valid_mask.device != device or kv_chunk_size_ptr.device != device:
        raise ValueError("prefill graph buffers and cache_seqlens must be on the same device")
    if kv_window_start_tokens.device != device or total_num_rows_ptr.device != device:
        raise ValueError("prefill graph scalar buffers and cache_seqlens must be on the same device")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if cta_tile_q <= 0:
        raise ValueError("cta_tile_q must be positive")
    if gqa_group_size <= 0:
        raise ValueError("gqa_group_size must be positive")
    if window_left < -1:
        raise ValueError("window_left must be -1 or non-negative")

    bs = int(batch)
    if bs <= 0:
        raise ValueError("prefill graph replay requires batch > 0")
    if int(cache_seqlens.shape[0]) < bs:
        raise ValueError("cache_seqlens is smaller than the graph batch")
    if int(cu_seqlens_q.shape[0]) < bs + 1:
        raise ValueError("cu_seqlens_q is smaller than the graph batch")
    work_items_capacity = int(request_indices.shape[0])
    block_valid_capacity = int(block_valid_mask.shape[0])
    if int(qo_tile_indices.shape[0]) != work_items_capacity or int(kv_tile_indices.shape[0]) != work_items_capacity:
        raise RuntimeError("prefill graph tile index buffers must match request_indices shape")
    if max_q_tiles_per_req <= 0:
        raise ValueError("max_q_tiles_per_req must be positive")
    if max_chunks_per_q_tile <= 0:
        raise ValueError("max_chunks_per_q_tile must be positive")
    if max_q_rows_per_req <= 0:
        raise ValueError("max_q_rows_per_req must be positive")
    required_work_items = bs * int(max_q_tiles_per_req) * int(max_chunks_per_q_tile)
    if required_work_items > work_items_capacity:
        raise RuntimeError(
            "prefill graph workspace request_indices capacity is too small for the graph plan"
        )
    if required_work_items > block_valid_capacity:
        raise RuntimeError(
            "prefill graph workspace block_valid capacity is too small for the graph plan"
        )
    if int(o_indptr.shape[0]) < bs + 1:
        raise RuntimeError("prefill graph workspace o_indptr capacity is too small")
    if int(kv_window_start_tokens.shape[0]) < bs:
        raise RuntimeError(
            "prefill graph workspace kv_window_start_tokens capacity is too small"
        )

    work_blocks = triton.cdiv(block_valid_capacity, _PREFILL_BLOCK_WORK_ITEMS)
    update_prefill_graph_work_metadata_triton[(work_blocks,)](
        cache_seqlens,
        cu_seqlens_q,
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        o_indptr,
        block_valid_mask,
        kv_chunk_size_ptr,
        kv_window_start_tokens,
        work_items_capacity=work_items_capacity,
        block_valid_capacity=block_valid_capacity,
        BATCH=bs,
        MAX_Q_TILES_PER_REQ=int(max_q_tiles_per_req),
        MAX_CHUNKS_PER_Q_TILE=int(max_chunks_per_q_tile),
        CTA_TILE_Q=int(cta_tile_q),
        GQA_GROUP_SIZE=int(gqa_group_size),
        PAGE_SIZE=int(page_size),
        WINDOW_LEFT=int(window_left),
        SPLIT_KV=bool(split_kv),
        BLOCK_WORK_ITEMS=_PREFILL_BLOCK_WORK_ITEMS,
    )
    torch.cumsum(
        o_indptr[1 : bs + 1],
        dim=0,
        out=o_indptr[1 : bs + 1],
    )
    total_num_rows_ptr[:1].copy_(cu_seqlens_q[bs : bs + 1])

    row_blocks = triton.cdiv(int(max_q_rows_per_req), _PREFILL_BLOCK_ROWS)
    update_prefill_graph_row_indptr_triton[(bs, row_blocks)](
        cu_seqlens_q,
        o_indptr,
        merge_indptr,
        BATCH=bs,
        MAX_Q_ROWS_PER_REQ=int(max_q_rows_per_req),
        BLOCK_ROWS=_PREFILL_BLOCK_ROWS,
    )
