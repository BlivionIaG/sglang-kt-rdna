# RESUME: Qwen3.8-Flash-Next-NVFP4 on par1-llm1

Branch `feat/qwen4-exp-support` @ BlivionIaG/sglang-kt-rdna. Rollback tag `pre-base-forward-2026-10-07`.
Status doc: `docs/qwen38-kt-port-status.md`. Run script: `scripts/run-qwen38-kt.sh`.

## THE PORT IS DONE. THE HOST IS THE BLOCKER.

Reached on the real checkpoint, in order: arg parsing -> model config -> architecture
resolution -> quantization dispatch -> ServerArgs construction -> Qwen2MoeSparseMoeBlock ->
KT CPU expert allocation -> **512 experts per layer, 30 of 48 MoE layers walked on 16 AMX
threads** -> then the host stops.

## ONE THING LEFT TO RUN (a single command, ~1 second)

Does the host-side gather's tensor path work on the REAL fp8 PLE table?

    cd ~/Projects/kt071-venv && source bin/activate
    python3 -c "
    import torch
    from safetensors import safe_open
    f = safe_open('/home/kletorch/models/qwen38-stage/model-fp8-mtp-ple.safetensors','pt')
    k = sorted(x for x in f.keys() if 'ngram_embedding' in x)[0]
    t = f.get_tensor(k); print(t.dtype, tuple(t.shape))
    out = t[torch.tensor([0,1,5,-1,t.shape[0]-1], dtype=torch.long)]
    print('indexed', out.dtype, tuple(out.shape))
    print('bf16', out.to(torch.bfloat16).dtype)"

The MASKING arithmetic is already verified against the Triton kernel's semantics (7/7 rows,
adversarial ids). Only these torch ops are unverified.

## THEN

    PLE_BACKEND=file ./scripts/run-qwen38-kt.sh     # sparse-mmap PLE + host-side gather
    # or, pinned, which upstream documents as NOT booting this model on a 92 GB box:
    # ./scripts/run-qwen38-kt.sh
    # add ISOLATE=1 to cap memory so a failure kills the load instead of wedging the host

## WHY `file` MATTERS

Upstream's own docstring (`models/qwen4_exp_ple_table.py`) says the `pinned` backend "does not
boot" this model at this size ("126.0 GiB of weights on a 121.63 GiB box"), and `pinned` is what
wedged par1-llm1 three times. The `file` backend streams a sparse mmap instead of pinning 47.7
GiB -- but its default gather needs unified memory, so this branch adds
`SGLANG_QWEN4_PLE_HOST_SIDE_GATHER=1` to read the rows on the CPU instead. That is the route
this host can actually take.

## HOST RULES (it flaps)

- ICMP/ARP fine, `:22` accepts TCP, banner never arrives, all service ports closed => nothing
  can fork; NOT a network fault, NOT a reboot (MAC unchanged).
- Recovery windows are SHORT. The first command in a window must be the one you want.
- Never hold a long foreground SSH call. Launch under tmux, read the log later.
- `chenco_adm@par1-cssec1` (mesh 100.122.94.247) is a working vantage on llm1's LAN and now
  trusts llm1's host key.
- No BMC on llm1; recovery from a wedge so far has been the stuck process exiting on its own.
