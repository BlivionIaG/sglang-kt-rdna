import importlib.util
import json
import os
import sys
import torch
from safetensors import safe_open

print(f"[exl3_ple] sglang_exl3_ple_patch MODULE LOADED by PID {os.getpid()}", flush=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_EXL3_PATH = os.path.join(_HERE, "exl3_ple_sglang.py")
if "exl3_ple_sglang" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("exl3_ple_sglang", _EXL3_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["exl3_ple_sglang"] = _mod
    _spec.loader.exec_module(_mod)
from exl3_ple_sglang import Exl3PackedNGramEmbedding


def _import_qwen4_exp_classes():
    """Import qwen4_exp lazily to avoid circular imports when sglang's
    import_model_classes is iterating models. qwen4_exp pulls in sglang.srt.configs
    and ultimately sglang.srt.models.transformers, which loops back to here.
    Defer until after the import-iteration is done."""
    from sglang.srt.models.qwen4_exp import (
        Qwen4ExpNGramEmbedding,
        Qwen4ExpPLELayer,
        Qwen4ExpLinearDecoderLayer,
        Qwen4ExpAttentionDecoderLayer,
    )
    return (
        Qwen4ExpNGramEmbedding,
        Qwen4ExpPLELayer,
        Qwen4ExpLinearDecoderLayer,
        Qwen4ExpAttentionDecoderLayer,
    )


def _is_exl3_pack(model_path: str) -> bool:
    idx_path = f"{model_path}/model.safetensors.index.json"
    if not os.path.exists(idx_path):
        return False
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx.get("weight_map", {})
    has_shards = any("shard_0.trellis" in k for k in wm)
    has_weight = any(k.endswith("ngram_embedding.weight") for k in wm)
    return has_shards and not has_weight


def _load_exl3_metadata(model_path: str) -> dict:
    with open(f"{model_path}/model.safetensors.index.json") as f:
        idx = json.load(f)
    sample_key = next(k for k in idx["weight_map"] if "shard_0.trellis" in k)
    sf_path = f"{model_path}/{idx['weight_map'][sample_key]}"
    layer_idx = int(sample_key.split(".")[3])
    with safe_open(sf_path, framework="pt", device="cpu") as f:
        base = f"model.language_model.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
        head_bias = f.get_tensor(f"{base}.head_bias")
        head_offsets = f.get_tensor(f"{base}.head_offsets").to(torch.int64)
        head_vocab_sizes = f.get_tensor(f"{base}.head_vocab_sizes").to(torch.int64)
        layer_multipliers = f.get_tensor(f"{base}.layer_multipliers").to(torch.int64)
    ngram_size = layer_multipliers.shape[0]
    H = head_offsets.shape[0]
    return {
        "head_bias": head_bias,
        "head_offsets": head_offsets,
        "head_vocab_sizes": head_vocab_sizes,
        "layer_multipliers": layer_multipliers,
        "ngram_size": ngram_size,
        "ngram_heads": H,
        "layer_idx": layer_idx,
    }


def _exl3_compute_ngram_ids(self, batch) -> torch.Tensor:
    """EXL3 hash matching sglang/sglang/kernels/ops/embeddings/qwen4_ngram.py:_qwen4_ngram_hash_kernel
    (and turboderp-org/exllamav3 exllamav3_ext/ngram.cu).

    The sglang call site is `compute_ngram_ids(batch)` where batch is a _PLEBatch. We replicate
    sglang's _hash_contexts extraction (decode fast path = direct ngram_context, otherwise
    unfold + index by req/token), then run the EXL3 hash.

    Returns: (bsz, ngram_heads=16) int64.
    """
    print(f"[exl3_ple] compute: batch.use_decode_fast_path={batch.use_decode_fast_path} ngram_context.shape={tuple(batch.ngram_context.shape) if batch.ngram_context is not None else None}", flush=True)
    if batch.use_decode_fast_path:
        contexts = batch.ngram_context
    else:
        contexts = batch.ngram_context.unfold(1, self.ngram_size, 1)[
            batch.req_indices, batch.token_offsets
        ]
    contexts = contexts.to(torch.long)
    meta = self.exl3_meta
    ngram_size = meta["ngram_size"]
    H = meta["ngram_heads"]
    heads_per_ngram = H // ngram_size
    mult_list = meta["layer_multipliers"].tolist()
    eos = int(self.eos_token_id)
    dev = contexts.device
    bsz, T = contexts.shape
    if T < ngram_size:
        out = torch.zeros(bsz, H, dtype=torch.int64, device=dev)
        for h in range(H):
            out[:, h] = meta["head_offsets"][h]
        return out
    t0 = contexts[:, 0]                                                      # oldest
    t1 = contexts[:, 1]
    t2 = contexts[:, 2]                                                      # current
    previous_1 = torch.where(t1 == eos, torch.full_like(t1, eos), t1)
    previous_2 = torch.where(
        (t0 == eos) | (t1 == eos),
        torch.full_like(t0, eos),
        t0,
    )
    mixed_bi = (t2 * mult_list[0]) ^ (previous_1 * mult_list[1])              # bigram
    mixed_tri = mixed_bi ^ (previous_2 * mult_list[2])                         # trigram
    out = torch.empty(bsz, H, dtype=torch.long, device=dev)
    for h in range(H):
        m = mixed_bi if h < heads_per_ngram else mixed_tri
        out[:, h] = (m % meta["head_vocab_sizes"][h]) + meta["head_offsets"][h]
    print(f"[exl3_ple] compute: out.shape={tuple(out.shape)} sample={out.flatten()[:8].tolist()}", flush=True)
    return out


def _exl3_embed_ngram_ids(self, ngram_ids: torch.Tensor, forward_batch, physical_tokens: int) -> torch.Tensor:
    print(f"[exl3_ple] embed: ngram_ids.shape={tuple(ngram_ids.shape)} range=[{ngram_ids.min().item()},{ngram_ids.max().item()}]", flush=True)
    lookup_ids, semantic_tokens = self._prepare_embedding_lookup(
        ngram_ids, forward_batch, physical_tokens
    )
    print(f"[exl3_ple] embed: lookup_ids.shape={tuple(lookup_ids.shape)} range=[{lookup_ids.min().item()},{lookup_ids.max().item()}]", flush=True)
    embeddings = self.exl3_embedding(lookup_ids)
    print(f"[exl3_ple] embed: out.shape={tuple(embeddings.shape)} |max|={embeddings.abs().max().item():.4f}", flush=True)
    out = self._finish_embedding_lookup(
        embeddings, semantic_tokens, forward_batch, physical_tokens
    )
    print(f"[exl3_ple] embed: final.shape={tuple(out.shape)} |max|={out.abs().max().item():.4f}", flush=True)
    return out


def _exl3_ngram_embedding_call(self, batch, forward_batch):
    ngram_ids = self.compute_ngram_ids(batch)
    print(f"[exl3_ple] fwd: ngram_ids.shape={tuple(ngram_ids.shape)} range=[{ngram_ids.min().item()},{ngram_ids.max().item()}] sample={ngram_ids.flatten()[:8].tolist()}", flush=True)
    embeddings = self._embed_ngram_ids(ngram_ids, forward_batch, batch.physical_tokens)
    return embeddings.flatten(start_dim=-2)


def patch_qwen4_exp_ple_to_exl3(model_path: str) -> None:
    if not _is_exl3_pack(model_path):
        for cand in (model_path, "/home/kletorch/models/qwen38-exl3", os.environ.get("PACK", "")):
            if cand and _is_exl3_pack(cand):
                model_path = cand
                break
        else:
            print(f"[exl3_ple] not EXL3 pack at {model_path} (no shard_0.trellis); no-op")
            return
    meta = _load_exl3_metadata(model_path)
    Qwen4ExpNGramEmbedding, Qwen4ExpPLELayer, Qwen4ExpLinearDecoderLayer, Qwen4ExpAttentionDecoderLayer = _import_qwen4_exp_classes()
    # Override ple_embed_dim to match the EXL3 pack (160, not 16*160)
    try:
        from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
        Qwen4ExpConfig.ple_embed_dim = 160
        Qwen4ExpTextConfig.ple_embed_dim = 160
    except Exception as e:
        print(f'[exl3_ple] could not override ple_embed_dim: {e}')

    _orig_init = Qwen4ExpNGramEmbedding.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # The EXL3 packed n-gram table is ~32 GB. Keep it CPU-pinned; gather
        # and dequant happen on CPU, then a small (B, H, dim) bf16 result is
        # moved to GPU on demand. Loading the full table onto the 16 GB GPU
        # OOMs during PLE init.
        self.exl3_embedding = Exl3PackedNGramEmbedding(
            model_path, meta["head_bias"], meta["layer_idx"]
        )
        with torch.no_grad():
            self.exl3_embedding.head_offsets = meta["head_offsets"].to(torch.int64).cpu()
            self.exl3_embedding.head_vocab_sizes = meta["head_vocab_sizes"].to(torch.int64).cpu()
        # Replace the VocabParallelEmbedding that the PLE prefetch path reads
        # via `self.ngram_embedding.gather(lookup_ids, out=...)`. The class
        # exposes the same .gather(lookup_ids, out=) interface that the model
        # calls in start_prefetch/_consume_prefetched_embeddings, so the
        # PLE's separate-stream prefetch + reduce path is reused unchanged.
        self.ngram_embedding = self.exl3_embedding
        with torch.no_grad():
            self.ngram_heads_offsets.data.copy_(meta["head_offsets"])
            self.ngram_heads_vocab_sizes.data.copy_(meta["head_vocab_sizes"])
            self.layer_multipliers.data.copy_(meta["layer_multipliers"])
        self.exl3_meta = meta
        self.ngram_size = meta["ngram_size"]
        self.ngram_heads = meta["ngram_heads"]

    Qwen4ExpNGramEmbedding.__init__ = _patched_init
    Qwen4ExpNGramEmbedding.compute_ngram_ids = _exl3_compute_ngram_ids
    Qwen4ExpNGramEmbedding._embed_ngram_ids = _exl3_embed_ngram_ids
    Qwen4ExpNGramEmbedding.forward = _exl3_ngram_embedding_call
    Qwen4ExpNGramEmbedding.is_exl3 = True

    _orig_init_qwen4_exp = Qwen4ExpPLELayer.__init__
    def _patched_ple_init(self, config, quant_config=None, prefix='', layer_id=None, ple_layer_index=0):
        _orig_init_qwen4_exp(self, config, quant_config, prefix, layer_id, ple_layer_index)
        for m in self.modules():
            if hasattr(m, 'in_features') and hasattr(m, 'out_features'):
                print(f'[exl3_ple] found linear: in={m.in_features} out={m.out_features}', flush=True)
                if m.in_features == 2560:
                    new_w = torch.nn.Parameter(m.weight[:, :160].contiguous())
                    m.in_features = 160
                    m.weight = new_w
                    print(f'[exl3_ple] trimmed linear to in={m.in_features}', flush=True)
    Qwen4ExpPLELayer.__init__ = _patched_ple_init
    _orig_init_linear = Qwen4ExpLinearDecoderLayer.__init__
    _orig_init_attn = Qwen4ExpAttentionDecoderLayer.__init__

    def _init_linear_traced(self, config, layer_id, quant_config=None, prefix="", alt_stream=None, is_nextn=False):
        _orig_init_linear(self, config, layer_id, quant_config, prefix, alt_stream, is_nextn)
        print(f"[exl3_ple] LinearDecoder init layer_id={layer_id} ple={getattr(self, 'ple', 'NA')}", flush=True)

    def _init_attn_traced(self, config, layer_id, quant_config=None, prefix="", alt_stream=None, is_nextn=False):
        _orig_init_attn(self, config, layer_id, quant_config, prefix, alt_stream, is_nextn)
        print(f"[exl3_ple] AttnDecoder init layer_id={layer_id} ple={getattr(self, 'ple', 'NA')}", flush=True)

    Qwen4ExpLinearDecoderLayer.__init__ = _init_linear_traced
    Qwen4ExpAttentionDecoderLayer.__init__ = _init_attn_traced
    print(f"[exl3_ple] patched Qwen4ExpLinearDecoderLayer.__init__ = {_init_linear_traced!r}", flush=True)
    print(f"[exl3_ple] patched Qwen4ExpAttentionDecoderLayer.__init__ = {_init_attn_traced!r}", flush=True)

    def _init_qwen4_exp_traced(self, config, quant_config=None, prefix="", layer_id=0, ple_layer_index=0):
        _orig_init_qwen4_exp(self, config, quant_config, prefix, layer_id, ple_layer_index)
        print(f"[exl3_ple] Qwen4ExpPLELayer.__init__: layer_id={layer_id} ple_layer_index={ple_layer_index} ple_module={self is not None} ngram_size={getattr(self, 'ple_embedding', None) and getattr(self.ple_embedding, 'ngram_size', 'NA')}", flush=True)

    Qwen4ExpPLELayer.__init__ = _init_qwen4_exp_traced

    _orig_ple_layer_forward = Qwen4ExpPLELayer.forward

    def _ple_layer_forward_traced(self, hidden_states, forward_batch, batch):
        print(f"[exl3_ple] PLE_LAYER.forward: hidden={tuple(hidden_states.shape)} batch.processed_tokens={getattr(batch, 'processed_tokens', 'NA')}", flush=True)
        return _orig_ple_layer_forward(self, hidden_states, forward_batch, batch)

    Qwen4ExpPLELayer.forward = _ple_layer_forward_traced
    print(f"[exl3_ple] patched Qwen4ExpNGramEmbedding for {model_path} (ngram_size={meta['ngram_size']}, heads={meta['ngram_heads']}, layer={meta['layer_idx']})")


if __name__ == "__main__":
    import sys
    pack = sys.argv[1] if len(sys.argv) > 1 else "/home/kletorch/models/qwen38-exl3"
    patch_qwen4_exp_ple_to_exl3(pack)
    print("patch applied (or no-op if not EXL3)")
else:
    # Auto-apply when sglang's import_model_classes iterates sglang.srt.models
    # and imports this file. This is the only path that reaches the worker
    # subprocesses (scheduler, detokenizer) where the model is actually built.
    # We hook sys.meta_path so the patch fires the moment qwen4_exp first imports
    # (i.e., right before sglang.srt.models.qwen4_exp's class objects are bound).
    # The class objects are patchable at that point; any later Qwen4ExpNGramEmbedding
    # instance will go through our patched __init__.

    class _Qwen4ExpImportHook:
        def find_spec(self, name, path=None, target=None):
            return None  # we never provide a spec ourselves

        def exec_module(self, module):
            pass

        def _fire_patch(self, name):
            if not name.startswith("sglang.srt.models.qwen4_exp"):
                return
            # First time qwen4_exp imports - apply patch now (synchronously, the
            # circular-import window has passed because transformers is already in
            # sys.modules by the time qwen4_exp starts importing).
            try:
                _auto_pack = os.environ.get("EXL3_PACK", "/home/kletorch/models/qwen38-exl3")
                if os.path.isdir(_auto_pack) and _is_exl3_pack(_auto_pack):
                    patch_qwen4_exp_ple_to_exl3(_auto_pack)
                    print(f"[exl3_ple] auto-patched in PID {os.getpid()} on {name}", flush=True)
            except Exception as _auto_err:
                print(f"[exl3_ple] auto-patch failed in PID {os.getpid()}: {_auto_err}", flush=True)
            # Remove this hook - one-shot.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass

        def find_module(self, name, path=None):
            # Deprecated but used by older loaders. Eagerly try the patch.
            self._fire_patch(name)
            return None

    _hook = _Qwen4ExpImportHook()

    # Wrap the standard import machinery: register a post-import hook by hooking
    # the importlib's _bootstrap._load hook. Simpler: use a custom finder that
    # triggers the patch on every import attempt of sglang.srt.models.qwen4_exp.
    class _PostImportFinder:
        def find_spec(self, name, path=None, target=None):
            if name == "sglang.srt.models.qwen4_exp" or name.startswith("sglang.srt.models.qwen4_exp."):
                # Schedule the patch to run after this import completes
                import importlib.abc
                class _Loader(importlib.abc.Loader):
                    def __init__(self, real_spec):
                        self.real_spec = real_spec
                    def create_module(self, spec):
                        return None
                    def exec_module(self, module):
                        # Run the real loader first
                        self.real_spec.loader.exec_module(module)
                        # Then patch (deferred because the module is now in sys.modules)
                        _hook._fire_patch(name)
                if target is None:
                    full_path = path
                else:
                    full_path = None
                from importlib.machinery import ModuleSpec
                return ModuleSpec(name, _Loader(None), origin=f"hooked:{name}")
            return None

    # Actually, the simpler approach: poll sys.modules for qwen4_exp in a background thread.
    import threading
    def _poll_for_qwen4():
        for _ in range(60):
            if "sglang.srt.models.qwen4_exp" in sys.modules:
                _hook._fire_patch("sglang.srt.models.qwen4_exp")
                return
            import time
            time.sleep(0.1)
        print(f"[exl3_ple] poll timed out waiting for qwen4_exp in PID {os.getpid()}", flush=True)

    threading.Thread(target=_poll_for_qwen4, daemon=True).start()
