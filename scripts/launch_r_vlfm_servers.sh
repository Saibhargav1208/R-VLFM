#!/usr/bin/env bash
# R-VLFM: Launch all model servers needed for R-VLFM evaluation
#
# Key difference from launch_vlm_servers.sh:
#   BLIP2ITM server is REPLACED by the Qwen2-VL-2B server
#
# Usage:
#   chmod +x scripts/launch_r_vlfm_servers.sh
#   ./scripts/launch_r_vlfm_servers.sh
#
# Then run evaluation:
#   python -m vlfm.run --config-name r_vlfm_objectnav_hm3d
#
# Requirements:
#   pip install transformers accelerate qwen-vl-utils
#   GPU with 8GB+ VRAM recommended

export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}
export SAM_PORT=${SAM_PORT:-12183}
export YOLOV7_PORT=${YOLOV7_PORT:-12184}

# ── R-VLFM: Qwen2-VL-2B replaces BLIP2ITM ──────────────────────────────────
export QWENVL_PORT=${QWENVL_PORT:-12190}
export QWENVL_MODEL=${QWENVL_MODEL:-"Qwen/Qwen2-VL-2B-Instruct"}
# ────────────────────────────────────────────────────────────────────────────

session_name=r_vlfm_servers_${RANDOM}

echo "========================================"
echo " R-VLFM Model Server Launcher"
echo "========================================"
echo " Qwen2-VL model : ${QWENVL_MODEL}"
echo " Qwen2-VL port  : ${QWENVL_PORT}"
echo " GroundingDINO  : ${GROUNDING_DINO_PORT}"
echo " MobileSAM      : ${SAM_PORT}"
echo " YOLOv7         : ${YOLOV7_PORT}"
echo "========================================"

# Create a detached tmux session with 4 panes
tmux new-session -d -s ${session_name}
tmux split-window -v -t ${session_name}:0
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2

# Pane 0: Qwen2-VL-2B server (replaces BLIP2ITM)
tmux send-keys -t ${session_name}:0.0 \
  "${VLFM_PYTHON} -m vlfm.vlm.qwen_vl --port ${QWENVL_PORT} --model ${QWENVL_MODEL}" C-m

# Pane 1: GroundingDINO (unchanged)
tmux send-keys -t ${session_name}:0.1 \
  "${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m

# Pane 2: MobileSAM (unchanged)
tmux send-keys -t ${session_name}:0.2 \
  "${VLFM_PYTHON} -m vlfm.vlm.sam --port ${SAM_PORT}" C-m

# Pane 3: YOLOv7 (unchanged)
tmux send-keys -t ${session_name}:0.3 \
  "${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m

echo ""
echo "Created tmux session '${session_name}'."
echo "Qwen2-VL-2B (~2B params) takes ~60-90s to load. Other models ~30s."
echo ""
echo "Monitor servers:"
echo "  tmux attach-session -t ${session_name}"
echo ""
echo "Once all servers are ready, run evaluation:"
echo "  python -m vlfm.run --config-name r_vlfm_objectnav_hm3d"
echo ""
echo "To kill all servers when done:"
echo "  tmux kill-session -t ${session_name}"
