#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "============================================"
echo "  club-3090 — Custom Presets Launcher"
echo "============================================"
echo ""
echo "  1) Artemis-31B-v1m   (uncensored Gemma fine-tune, beellama dual, 131K)
  echo "  2) Gembrain-X-31B    (12+ RP merge, uncensored creative, beellama dual, 131K)""
echo "  2) Gemma-4-E4B Q4_0  (lightweight, 8-stream, vision, beellama dual)"
echo "  3) Gemma-4-E4B bf16  (vLLM dual, 8-stream, 64K, untested)"
echo "  4) Qwen 3.6 27B      (vLLM dual, fp8-mtp, 262K — upstream default)"
echo "  5) Gemma 4 31B       (vLLM dual, bf16-mtp, 262K — upstream default)"
echo ""
echo "  s) Status — show running containers"
echo "  x) Stop all"
echo "  q) Quit"
echo ""

read -rp "Pick [1-5/s/x/q]: " choice

case "$choice" in
  1) bash scripts/switch.sh --force beellama/artemis-dual-dflash ;;
  2) bash scripts/switch.sh --force beellama/gembrain-x-dual-q4km ;;
  3) bash scripts/switch.sh --force beellama/gemma-e4b-dual-q4_0 ;;
  4) bash scripts/switch.sh --force vllm/gemma-e4b-dual-bf16 ;;
  5) bash scripts/switch.sh vllm/dual ;;
  6) bash scripts/switch.sh vllm/gemma-31b-dual ;;
  s) docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | head -1
     docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E "vllm|beellama|llama" || echo "  (none running)" ;;
  x) docker ps --format '{{.Names}}' | grep -E "vllm|beellama|llama" | xargs -r docker stop
     echo "Stopped." ;;
  q) exit 0 ;;
  *) echo "Invalid choice." ; exit 1 ;;
esac
