# R-VLFM: Relational Vision-Language Frontier Maps
#
# Extends ITMPolicyV2 with two changes:
#   1. Replace BLIP-2 with Qwen2-VL-2B for frontier scoring (Stage 1)
#   2. Add relational verification on top-K frontiers (Stage 2) — new contribution
#
# Usage: drop-in replacement for ITMPolicyV2 in config files.
# Set env var QWENVL_PORT (default 12190) to point at the QwenVL server.

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch import Tensor

from vlfm.policy.itm_policy import BaseITMPolicy, ITMPolicyV2, PROMPT_SEPARATOR
from vlfm.vlm.qwen_vl import QwenVLClient

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

# ── Hyperparameters ──────────────────────────────────────────────────────────

# How many top Stage-1 frontiers to re-verify with Stage-2 relational check
TOP_K_VERIFY = 3

# Minimum Stage-1 score for a frontier to be considered for Stage-2 verification
STAGE1_MIN_SCORE = 0.20

# Weight blending Stage-1 and Stage-2 scores: final = α*s1 + (1-α)*s2
STAGE1_WEIGHT = 0.4
STAGE2_WEIGHT = 0.6

# ─────────────────────────────────────────────────────────────────────────────


class RVLFMPolicy(ITMPolicyV2):
    """
    R-VLFM: Relational Vision-Language Frontier Maps

    Key changes over vanilla VLFM (ITMPolicyV2):

    Stage 1 — Replace BLIP-2 with Qwen2-VL-2B:
        value_map.update() now calls QwenVL.score_frontier(image, object_name)
        instead of BLIP2ITM.cosine(image, prompt).
        The value map logic is otherwise identical.

    Stage 2 — Relational Verification (new contribution):
        After Stage-1 sorts all frontiers, take the top-K frontiers.
        For each, retrieve the cached RGB observation nearest to that frontier.
        Ask Qwen2-VL: "Is there [relational_goal] in this image?" → float.
        Final score = STAGE1_WEIGHT * s1 + STAGE2_WEIGHT * s2.
        Only triggered when the goal contains a spatial relation
        (e.g. "cup on the kitchen counter"). Falls back to pure Stage-1
        for simple goals (e.g. "cup").

    Goal parsing:
        Calls QwenVL.parse_goal() once at episode start to extract:
            object_name:    "cup"          (used in Stage 1)
            relational_goal: "a cup on the kitchen counter" (used in Stage 2)
            is_relational:   True/False
    """

    def __init__(
        self,
        *args: Any,
        qwenvl_port: int = int(os.environ.get("QWENVL_PORT", "12190")),
        **kwargs: Any,
    ) -> None:
        # Call parent __init__ — this sets self._itm = BLIP2ITMClient(...)
        super().__init__(*args, **kwargs)

        # Replace BLIP-2 with Qwen-VL (drop-in: has .cosine() method)
        self._itm = QwenVLClient(port=qwenvl_port)
        self._qwen = self._itm  # alias for clarity when calling Stage-2 methods

        # Will be populated when _parse_goal() is called in _reset()
        self._object_name: str = ""
        self._relational_goal: str = ""
        self._is_relational: bool = False

        # Cache: maps quantized map coordinate → (rgb_image, stage1_score)
        # Used to retrieve frontier-proximal RGB frames at verification time
        self._frontier_obs_cache: Dict[Tuple[int, int], Tuple[np.ndarray, float]] = {}

        print("[R-VLFM] Initialized. Qwen2-VL-2B replaces BLIP-2.")
        print(f"[R-VLFM] Stage-2 relational verification on top-{TOP_K_VERIFY} frontiers.")

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        super()._reset()
        self._frontier_obs_cache.clear()
        # Re-parse goal when episode resets
        self._parse_and_store_goal(self._target_object)

    # ── Goal parsing ─────────────────────────────────────────────────────────

    def _parse_and_store_goal(self, goal_text: str) -> None:
        """
        Parse the navigation goal into structured components.

        For "the red cup on the kitchen counter":
            object_name    = "cup"
            relational_goal = "a red cup on the kitchen counter"
            is_relational  = True

        For "chair":
            object_name    = "chair"
            relational_goal = ""
            is_relational  = False
        """
        if not goal_text:
            return

        # Clean up goal text (remove VLFM pipe separators if present)
        clean_goal = goal_text.replace("|", "/").strip()

        try:
            parsed = self._qwen.parse_goal(clean_goal)
            self._object_name = parsed.get("object", clean_goal)
            self._is_relational = bool(parsed.get("is_relational", False))
            relation = parsed.get("relation", "")

            if self._is_relational and relation:
                self._relational_goal = f"a {self._object_name} {relation}"
            else:
                # Simple goal — Stage 2 will be skipped
                self._relational_goal = self._object_name
                self._is_relational = False

        except Exception as e:
            # Fallback: treat entire goal text as simple object name
            print(f"[R-VLFM] Goal parsing failed ({e}), using raw text: {clean_goal}")
            self._object_name = clean_goal
            self._relational_goal = clean_goal
            self._is_relational = False

        print(f"[R-VLFM] Goal parsed:")
        print(f"         object_name     = '{self._object_name}'")
        print(f"         relational_goal = '{self._relational_goal}'")
        print(f"         is_relational   = {self._is_relational}")

    # ── Value map update (Stage 1) ───────────────────────────────────────────

    def _update_value_map(self) -> None:
        """
        Overrides BaseITMPolicy._update_value_map().

        Uses self._object_name (not the full relational text) for Stage-1
        frontier scoring. This scores: "does this scene suggest a CUP nearby?"
        rather than "does this scene have a cup on a kitchen counter?"
        (that precision is handled by Stage-2 verification).

        Also updates self._frontier_obs_cache with (rgb, score) pairs
        keyed by quantized robot position, for use in Stage-2 verification.
        """
        all_rgbd = self._observations_cache["value_map_rgbd"]
        robot_xy = self._observations_cache["robot_xy"]

        # Build scores using object_name only (Stage 1)
        scores = []
        for rgb, depth, tf, min_depth, max_depth, fov in all_rgbd:
            # Score how likely this image leads to the target object
            s = self._itm.cosine(rgb, self._object_name)
            scores.append([s])

            # Cache this observation for Stage-2 retrieval
            # Key: robot position quantized to 0.5m grid cells
            key = (int(robot_xy[0] / 0.5), int(robot_xy[1] / 0.5))
            self._frontier_obs_cache[key] = (rgb.copy(), s)

        # Update value map (identical to original logic)
        for score_vec, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            scores, all_rgbd
        ):
            self._value_map.update_map(
                np.array(score_vec), depth, tf, min_depth, max_depth, fov
            )

        self._value_map.update_agent_traj(
            robot_xy,
            self._observations_cache["robot_heading"],
        )

    # ── Stage-2: relational verification ─────────────────────────────────────

    def _get_frontier_rgb(self, frontier_xy: np.ndarray) -> Optional[np.ndarray]:
        """
        Retrieve the cached RGB image spatially closest to frontier_xy.
        Returns None if cache is empty.
        """
        if not self._frontier_obs_cache:
            return None

        fq = np.array([frontier_xy[0] / 0.5, frontier_xy[1] / 0.5])
        best_key = min(
            self._frontier_obs_cache.keys(),
            key=lambda k: np.linalg.norm(np.array(k) - fq),
        )
        rgb, _ = self._frontier_obs_cache[best_key]
        return rgb

    def _stage2_verify(self, frontier_xy: np.ndarray) -> float:
        """
        Run Stage-2 relational verification for a single frontier.

        Retrieves the cached RGB nearest to this frontier and asks Qwen:
        "Is there [relational_goal] in this image?" → float [0, 1]
        """
        rgb = self._get_frontier_rgb(frontier_xy)
        if rgb is None:
            # No cached observation — return neutral score
            return 0.5

        try:
            score = self._qwen.verify_relation(rgb, self._relational_goal)
            return score
        except Exception as e:
            print(f"[R-VLFM] Stage-2 verify failed: {e}")
            return 0.5

    # ── Best frontier selection (Stage 1 + Stage 2) ──────────────────────────

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Two-stage frontier selection:

        Stage 1 (inherited): Sort all frontiers by value map score.
        Stage 2 (new): Verify top-K frontiers with Qwen relational VQA.
                       Only active when goal is relational (e.g. "cup on desk").

        Returns the frontier with the highest combined score.
        """
        # ── Stage 1: get sorted frontiers from value map (original VLFM logic)
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)

        if not self._is_relational:
            # Simple goal (e.g. "chair") — no Stage-2 needed
            # Fall through to original _get_best_frontier logic
            return super()._get_best_frontier(observations, frontiers)

        # ── Stage 2: relational verification on top-K candidates ─────────────

        # Candidate pool: top-K frontiers with Stage-1 score above threshold
        candidates = [
            (pt, s1)
            for pt, s1 in zip(sorted_pts, sorted_values)
            if s1 >= STAGE1_MIN_SCORE
        ][:TOP_K_VERIFY]

        if not candidates:
            # All Stage-1 scores below threshold — fall back to Stage-1 only
            print("[R-VLFM] All Stage-1 scores below threshold, skipping Stage-2.")
            return super()._get_best_frontier(observations, frontiers)

        print(f"[R-VLFM] Running Stage-2 on top-{len(candidates)} candidates...")

        # Compute combined scores
        combined: List[Tuple[np.ndarray, float]] = []
        for pt, s1 in candidates:
            s2 = self._stage2_verify(pt)
            final_score = STAGE1_WEIGHT * s1 + STAGE2_WEIGHT * s2
            print(
                f"[R-VLFM]   frontier {pt} | "
                f"s1={s1:.3f} | s2={s2:.3f} | combined={final_score:.3f}"
            )
            combined.append((pt, final_score))

        # Sort by combined score descending
        combined.sort(key=lambda x: x[1], reverse=True)

        best_frontier, best_score = combined[0]

        os.environ["DEBUG_INFO"] = (
            f"R-VLFM Stage2 best: {best_score*100:.1f}% "
            f"(goal: '{self._relational_goal}')"
        )
        print(f"[R-VLFM] Best frontier selected: {best_frontier} | score={best_score:.3f}")

        self._last_frontier = best_frontier
        self._last_value = best_score

        return best_frontier, best_score

    # ── Policy info (adds R-VLFM debug info to visualization) ────────────────

    def _get_policy_info(self, detections: Any) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)
        policy_info["r_vlfm_goal"] = {
            "object": self._object_name,
            "relation": self._relational_goal,
            "is_relational": self._is_relational,
        }
        return policy_info
