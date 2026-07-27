# Gemma 4 E4B — on 2× RTX 3090

**Run [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) — the lightest model in the
Gemma 4 family (~8B raw / ~4.5B effective params, dense, vision + audio + tool calling) — on
2× RTX 3090s, tuned for a multi-tenant agent fleet (8 concurrent streams).**

> 🐣 **Incubating.** This catalog entry was authored from architecture facts (HF `config.json` +
> model card) and hand-derived VRAM math — **not yet booted on real hardware**. No verify-full,
> no bench, no soak, no calibration. See "What's unvalidated" below before treating any number
> here as measured. Promote to 🧪 Experimental / ⚠️ / ✅ once `docs/ADDING_MODELS.md` Steps 5-8
> run for real on a dual-3090 Linux rig.

---

## Deployment

```bash
bash scripts/launch.sh --variant vllm/gemma-e4b-dual-bf16 --force   # --force required: incubating status
```

| Config | Max ctx | Concurrency | Best for |
|--------|---------|-------------|----------|
| `vllm/gemma-e4b-dual-bf16` | 65536 (64K) / stream | **8 concurrent streams** | Multi-tenant agent fleet — vision + tools, lightest Gemma 4 member |

`switch.sh --list --all` will show it (hidden from the default `--list` while incubating).

## Models

- **Target:** [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) (bf16, ~14.9 GB — the ONLY shipped format)
- **Draft (unused today):** [`google/gemma-4-E4B-it-assistant`](https://huggingface.co/google/gemma-4-E4B-it-assistant) (~0.16 GB MTP head) — exists upstream but MTP is shipped disabled (see Gotchas)

No Intel AutoRound / cyankiwi AWQ quant exists for E4B yet (confirmed via authenticated HF API
lookup — both 404 as of this writing). Not urgent: bf16 weights are only ~7.45 GB/card at TP=2.

## Key details

| Aspect | Notes |
|--------|-------|
| **Arch** | Dense, `gemma4-swa-dense` family (SAME family as 31B — NOT the 12B "unified"/encoder-free arch). 42 layers: 7 full-attention + 35 sliding (window=512). |
| **Quants** | bf16 only (see above) |
| **KV** | bf16 only shipped; this model's `attention_k_eq_v` is **`false`** — the one Gemma-4 family member with UNTIED K/V (every sibling — 12B/26B-A4B/31B — is tied). Don't reuse KV-sizing intuition from those. |
| **Vision** | ✅ yes (+ audio — no special compose flags needed, matches 12B/31B convention) |
| **Tools** | ✅ `--tool-call-parser gemma4` (native ParserEngine on vllm-stable v0.25.1) |
| **MTP** | Drafter exists upstream but is **shipped disabled** — Gemma-4 MTP × tool-calling is broken on vllm-stable (vLLM #39043; fix #42006 closed-unmerged), same family-wide issue affecting `gemma-31b-dual` / `gemma-4-12b-dual-mtp`. Re-enable snippet is commented at the bottom of the compose. |
| **Context ceiling** | 131072 (128K) — HALF of 12B/26B-A4B/31B's 256K. Don't assume the family default. |
| **NVLink** | Auto-detected via `NVLINK_MODE` env var (same as siblings) |

## Why 8 concurrent streams @ 64K instead of one long-context stream

This model is dramatically lighter than the rest of the Gemma 4 family (num_kv_heads=2 vs 31B's
16; ~7.45 GB/card weights at TP=2 vs 31B's 9-29 GB/card), so the natural way to spend that
headroom is concurrency, not depth — a small fleet of agents/tools/chat sessions sharing one
dual-3090 box, rather than one giant single-stream context. If your workload is the opposite
(one deep conversation, not many shallow ones), override `MAX_NUM_SEQS=1` and raise
`MAX_MODEL_LEN` toward the real 131072 ceiling — see "KV math" in the compose header for the
projected headroom.

## What's unvalidated (read before trusting a number here)

This entry was built on a Windows dev workstation with no access to real dual-RTX-3090 Linux
hardware, so unlike every other model in this catalog:

- **No boot log.** All VRAM/KV sizing in the compose header is hand-derived from
  `config.json` + the shared Gemma-4 family formula (see `docs/KV_MATH.md`), not measured.
- **`kv-calc.py` does not model this spec** (`kv_calc_supported: false` in the profile YAML).
  Reason: kv-calc's `gemma4-swa-dense` branch hardcodes `k_v_tensors=1` (K==V tied) for every
  Gemma-4 family member it knows about — correct for 12B/26B-A4B/31B, but this model's
  `attention_k_eq_v` is `false` (untied). Naively running `kv-calc.py --model gemma-4-e4b`
  would silently under-predict KV usage by 2× — the compose's hand-derived numbers already
  apply `k_v_tensors=2` correctly, but they're still a projection, not a measurement.
- **No verify-full / verify-stress / bench / soak / quality-test.** None of the usual gates
  have run.
- **Shipped `max-model-len=65536` at `max-num-seqs=8`** uses only ~61% of the *projected* KV
  pool (per the compose header's math) as a safety margin against the above being an estimate.
  The theoretical ceiling at 8 concurrent full-length streams is ~100-106K tokens/stream —
  worth trying once a real boot log confirms the actual `Available KV cache` line.

**Before promoting past 🐣 Incubating**, run on real hardware (`docs/ADDING_MODELS.md` Steps 5-8):

```bash
bash scripts/launch.sh --variant vllm/gemma-e4b-dual-bf16 --force
docker logs <container> 2>&1 | grep -iE "kv cache|model length|gpu memory|allocated" \
  | tee /tmp/gemma-4-e4b-boot.log
bash scripts/verify-full.sh
bash scripts/verify-stress.sh
bash scripts/bench.sh
```

Then back-solve `per_token_bytes` from the boot log against the KV math in the compose header —
if it's roughly in line, the shipped `65536 @ seqs=8` can likely be raised; if it's off by ~2×,
suspect the K=V-tying assumption got flipped somewhere (see `docs/KV_MATH.md` "Common pitfalls").

## Gotchas

1. **Untied KV is model-specific, not family-wide.** If you're extending this catalog entry
   (new quant, new topology), re-derive KV math from scratch — don't copy the 31B/26B-A4B/12B
   `k_v_tensors=1` formula.
2. **`kv-calc.py --model gemma-4-e4b` doesn't exist** (not wired into the tool's `MODEL_SPECS` —
   see the profile YAML comment). Use the hand-derived math in the compose header, or wire a
   proper `gemma4-swa-dense`-untied branch into `tools/kv-calc.py` once there's a real
   calibration anchor to tune the activation coefficient against.
3. **MTP is disabled on purpose**, not an oversight — see "Key details" above. Don't re-enable
   without first checking `vLLM #39043` / `#42006` have landed on a stable pin.
4. **128K ceiling, not 256K** — this is the one Gemma-4 family member with the smaller context
   window (matches the E2B/E4B row in the model card, not the 12B/26B-A4B/31B row).

## Upstream tracker

- [`docs/UPSTREAM.md`](../../docs/UPSTREAM.md) — Gemma-4 MTP × tool-calling (vLLM #39043 /
  #42006) is the row that governs this model's MTP-disabled status too.
