"""Backward-compatible shim -- all code has moved to sglang.srt.utils.hf_transformers.

Converted from the old 869-line module when the qwen4 subsystem was ported: that
copy carried its own `_CONFIG_REGISTRY` of 33 entries and called `AutoConfig.register`
for each, which is why `--model-path` on a qwen4_exp checkpoint failed with
"The checkpoint you are trying to load has model type `qwen4_exp` but Transformers does
not recognize this architecture" even though `hf_transformers/common.py` already
registered it -- the server's `get_config` came from THIS module's registry, which never
had the entry. Delegating to `hf_transformers` means one registry, the 70-name one.

Every name imported from this module elsewhere in the tree is re-exported here; verified
before the conversion that none of the 8 consumers (AutoConfig, check_gguf_file,
download_from_hf, get_config, get_processor, get_rope_config, get_tokenizer,
resolve_hf_gguf_reference) would break.
"""
from __future__ import annotations

from sglang.srt.utils.hf_transformers import *  # noqa: F401, F403
from sglang.srt.utils.hf_transformers import __all__  # noqa: F401
