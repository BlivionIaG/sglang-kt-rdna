import mmap
import os
import json
"""sglang patch: replace Qwen4ExpNGramEmbedding's VocabParallelEmbedding with an
EXL3-packed table lookup when the on-disk format is the EXL3 int16-trellis shards.

Apply this patch to <venv>/lib/python3.12/site-packages/sglang/srt/models/qwen4_exp.py
by replacing the `Qwen4ExpNGramEmbedding` class. The packed table is loaded
once at init time (one-time 24s for 32 GB on first model load), then each forward
pass hashes the n-gram IDs, gathers the packed rows, and dequantizes to bf16.
"""
import torch
from safetensors import safe_open

MUL1 = 0x83DCD12D
ROW_DIM = 160
NUM_SHARDS = 128
SHARD_COLS = 51
NGRAM_VOCAB_PER_SHARD = 2500012
EOS_TOKEN_DEFAULT = 248044
K_DEFAULT = 5


def _mul1_codebook(device):
    s = torch.arange(65536, dtype=torch.int64, device=device)
    prod = (s * MUL1) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    h = (1024 + bsum).float()
    k_inv = torch.tensor([0x1eee], dtype=torch.uint16).view(torch.float16).float().item()
    k_bias = torch.tensor([0xc931], dtype=torch.uint16).view(torch.float16).float().item()
    return (h * k_inv + k_bias).to(torch.float16)


def _ring_dim(words, K):
    return (words - 1) * 16 // K


def _unpack(packed, K):
    dev = packed.device
    words = packed.shape[1] - 1
    dim = _ring_dim(packed.shape[1], K)
    scales = packed[:, 0].contiguous().view(torch.float16)
    stream = packed[:, 1:].contiguous().view(torch.uint16).to(torch.int64)
    bit = torch.arange(dim, device=dev) * K
    window = stream[:, bit >> 4] | (stream[:, ((bit >> 4) + 1) % words] << 16)
    symbols = (window >> (bit & 15)) & ((1 << K) - 1)
    states = torch.zeros_like(symbols)
    for j in range((15 + K) // K):
        states |= torch.roll(symbols, j, dims=1) << (j * K)
    return states & 0xFFFF, scales


def _hadamard(x, block):
    shape = x.shape
    n = shape[-1]
    if n != block:
        xx = x.reshape(*shape[:-1], n // block, block)
        h = 1
        while h < block:
            xx = xx.view(*shape[:-1], n // block, block // (2 * h), 2, h)
            a, b = xx[..., 0, :], xx[..., 1, :]
            xx = torch.stack((a + b, a - b), dim=-2)
            h *= 2
        return xx.reshape(shape) * (block ** -0.5)
    xx = x.reshape(-1, n)
    h = 1
    while h < n:
        xx = xx.view(-1, n // (2 * h), 2, h)
        a, b = xx[:, :, 0], xx[:, :, 1]
        xx = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return (xx.reshape(shape) * (n ** -0.5))


def _dequant(packed, K=5):
    states, scales = _unpack(packed, K)
    out = _mul1_codebook(packed.device)[states].float() * scales.float().unsqueeze(1)
    return _hadamard(out, 16).to(torch.bfloat16)


class Exl3PackedNGramEmbedding(torch.nn.Module):
    """Drop-in replacement for VocabParallelEmbedding when the PLE table is in
    the EXL3 int16-trellis packed format. The forward(path) takes a (bsz, total_heads)
    int64 of row indices (computed by compute_ngram_ids) and returns (bsz, total_heads, dim) bf16.
    """

    def __init__(self, pack_path: str, head_bias: torch.Tensor, layer_idx: int, K: int = K_DEFAULT):
        super().__init__()
        self.K = K
        self.register_buffer("head_bias", head_bias)              # (H, dim) fp16
        # mmap the 32 GB EXL3 ngram_embedding.safetensors file as a single
        # contiguous int16 tensor (320_001_536, 51). The shards are already in
        # shard-0..shard-127 order in the file. mmap means the page is only
        # loaded into RAM on first access (UVA read).
        import numpy as np
        _PATH = f"{pack_path}/ngram_embedding.safetensors"
        with open(_PATH, "rb") as _fp:
            _hdr_size = int.from_bytes(_fp.read(8), "little")
            _hdr = json.loads(_fp.read(_hdr_size))
        _trellis_keys = sorted([k for k in _hdr if k.endswith(".trellis")],
                             key=lambda k: int(k.split("_")[-1].split(".")[0]))
        _off0 = _hdr[_trellis_keys[0]]["data_offsets"][0] + 8 + _hdr_size
        _offN = _hdr[_trellis_keys[-1]]["data_offsets"][1] + 8 + _hdr_size
        _total = NUM_SHARDS * 2500012 * SHARD_COLS
        _total_bytes = _total * 2
        assert _offN - _off0 == _total_bytes, f"mmap size mismatch: {_offN - _off0} vs {_total_bytes}"
        _fd = os.open(_PATH, os.O_RDONLY)
        _mm = mmap.mmap(_fd, 0, prot=mmap.PROT_READ)
        _arr = np.frombuffer(_mm, dtype=np.int16, offset=_off0, count=_total).reshape(NUM_SHARDS * 2500012, SHARD_COLS)
        stacked = torch.from_numpy(_arr)
        self.register_buffer("packed", stacked, persistent=False)
        # Some paths (qwen4_exp._consume_prefetched_embeddings) read
        # ``self.ple_embedding.ngram_embedding.weight_scale`` and multiply
        # the gathered embeddings by it. The original VocabParallelEmbedding
        # set this to a ones(1) bf16 buffer; the EXL3 pack is unscaled so
        # we mirror that contract with an explicit ones(1) bf16 buffer.
        self.register_buffer("weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True)
        self._mmap_handle = _mm
        self.head_offsets = None
        self.head_vocab_sizes = None
        self._mmap_fd = _fd
        self.dim = _ring_dim(SHARD_COLS, K)

    def allocate_output(self, shape, device):
        # The model's _allocate_prefetch_buffer calls this with (lookup_tokens, ple_embed_dim=2560).
        # For the EXL3 pack the effective per-token dim is 16*self.dim = 2560, with
        # lookup_tokens being B*16 (B tokens, 16 heads per token). Override to allocate
        # (B, 2560) so the buffer matches the gather output after the 16-head broadcast.
        lookup_tokens = shape[0]
        H = self.head_bias.shape[0]
        if lookup_tokens % H == 0:
            B = lookup_tokens // H
        else:
            B = lookup_tokens
        return torch.empty((B, H * self.dim), dtype=torch.bfloat16, device=device)

    def reduce(self, output):
        return output

    def gather(self, row_idx, out):
        N = row_idx.numel()
        if N == 0:
            return out
        H = self.head_bias.shape[0]
        if N % H != 0:
            B = N
        else:
            B = N // H
        if not hasattr(self, '_row_buf') or self._row_buf.numel() < N:
            self._row_buf = torch.empty(N, dtype=torch.int64, pin_memory=True)
            self._deq_buf = torch.empty(N, self.dim, dtype=torch.bfloat16, pin_memory=True)
        row_buf = self._row_buf[:N]
        deq_buf = self._deq_buf[:N]
        flat = row_idx.reshape(-1)
        if flat.device.type == 'cuda':
            flat_cpu = flat.to(torch.int64, non_blocking=True).cpu()
        else:
            flat_cpu = flat.to(torch.int64)
        row_buf.copy_(flat_cpu, non_blocking=True)
        rows = self.packed[row_buf]
        deq = _dequant(rows, K=self.K)
        head_idx = torch.arange(N) % H
        torch.add(deq, self.head_bias[head_idx], out=deq_buf)
        if B != N:
            deq_buf = deq_buf.contiguous().reshape(B, H * self.dim)
            print(f'[exl3_gather] deq_buf reshaped to {deq_buf.shape}, out shape {out.shape}', flush=True)
        out.copy_(deq_buf, non_blocking=True)
        return out


    def forward(self, row_idx):
        if row_idx.numel() == 0:
            return torch.empty((0, self.head_bias.shape[0], self.dim), dtype=torch.bfloat16, device=row_idx.device)
        n = row_idx.numel()
        out = self.allocate_output((n, self.head_bias.shape[0], self.dim), row_idx.device)
        return self.gather(row_idx, out)
