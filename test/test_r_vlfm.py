# R-VLFM tests
# Tests for QwenVL utilities and RVLFMPolicy logic.
# All Qwen model calls are mocked — no GPU required.

import json
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────
# Helper: make a dummy RGB image
# ─────────────────────────────────────────────────────────────

def _dummy_rgb(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────
# Tests for _extract_float (no model needed)
# ─────────────────────────────────────────────────────────────

class TestExtractFloat:
    """Unit tests for the score extraction helper in qwen_vl.py."""

    def _extract(self, text: str) -> float:
        from vlfm.vlm.qwen_vl import _extract_float
        return _extract_float(text)

    def test_plain_decimal(self) -> None:
        assert self._extract("0.85") == pytest.approx(0.85)

    def test_decimal_in_sentence(self) -> None:
        assert self._extract("The score is 0.72 out of 1.") == pytest.approx(0.72)

    def test_integer_response(self) -> None:
        # Model says "1" — should clamp to 1.0
        assert self._extract("1") == pytest.approx(1.0)

    def test_clamp_above_one(self) -> None:
        # Model hallucinates "1.5" — must clamp to 1.0
        assert self._extract("1.5") == pytest.approx(1.0)

    def test_clamp_below_zero(self) -> None:
        # Negative value — clamp to 0.0
        assert self._extract("-0.3") == pytest.approx(0.0)

    def test_no_number_returns_zero(self) -> None:
        assert self._extract("I cannot determine this.") == pytest.approx(0.0)

    def test_empty_string(self) -> None:
        assert self._extract("") == pytest.approx(0.0)

    def test_percentage_style(self) -> None:
        # "72%" — extracts 72, clamps to 1.0 (since >1)
        val = self._extract("72%")
        assert 0.0 <= val <= 1.0


# ─────────────────────────────────────────────────────────────
# Tests for _extract_json (no model needed)
# ─────────────────────────────────────────────────────────────

class TestExtractJson:
    """Unit tests for the JSON extraction helper in qwen_vl.py."""

    def _extract(self, text: str) -> Dict:
        from vlfm.vlm.qwen_vl import _extract_json
        return _extract_json(text)

    def test_clean_json(self) -> None:
        text = '{"object": "cup", "relation": "on the counter", "is_relational": true}'
        result = self._extract(text)
        assert result["object"] == "cup"
        assert result["relation"] == "on the counter"
        assert result["is_relational"] is True

    def test_json_in_sentence(self) -> None:
        text = 'Here is the result: {"object": "chair", "relation": "", "is_relational": false}'
        result = self._extract(text)
        assert result["object"] == "chair"
        assert result["is_relational"] is False

    def test_simple_object_fallback(self) -> None:
        # Model returns plain text instead of JSON
        result = self._extract("chair")
        assert result["object"] == "chair"
        assert result["relation"] == ""
        assert result["is_relational"] is False

    def test_relational_goal(self) -> None:
        payload = {
            "object": "laptop",
            "relation": "near the whiteboard",
            "is_relational": True,
        }
        result = self._extract(json.dumps(payload))
        assert result["object"] == "laptop"
        assert result["relation"] == "near the whiteboard"
        assert result["is_relational"] is True


# ─────────────────────────────────────────────────────────────
# Tests for QwenVL (model mocked)
# ─────────────────────────────────────────────────────────────

class TestQwenVLMocked:
    """Tests for QwenVL methods with the model._run() mocked out."""

    def _make_qwen(self, run_return: str) -> "QwenVL":
        """Create a QwenVL instance with _run() mocked."""
        from vlfm.vlm.qwen_vl import QwenVL
        qwen = QwenVL.__new__(QwenVL)  # skip __init__ (no model loaded)
        qwen.max_new_tokens = 32
        qwen._run = MagicMock(return_value=run_return)
        return qwen

    def test_score_frontier_returns_float(self) -> None:
        qwen = self._make_qwen("0.73")
        score = qwen.score_frontier(_dummy_rgb(), "cup")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.73)

    def test_score_frontier_calls_run_once(self) -> None:
        qwen = self._make_qwen("0.5")
        qwen.score_frontier(_dummy_rgb(), "chair")
        qwen._run.assert_called_once()

    def test_verify_relation_returns_float(self) -> None:
        qwen = self._make_qwen("0.91")
        score = qwen.verify_relation(_dummy_rgb(), "a cup on the kitchen counter")
        assert isinstance(score, float)
        assert score == pytest.approx(0.91)

    def test_verify_relation_prompt_contains_goal(self) -> None:
        qwen = self._make_qwen("0.5")
        goal = "a laptop near the whiteboard"
        qwen.verify_relation(_dummy_rgb(), goal)
        call_args = qwen._run.call_args
        prompt_used = call_args[0][1]  # second positional arg to _run
        assert goal in prompt_used

    def test_cosine_is_alias_for_score_frontier(self) -> None:
        qwen = self._make_qwen("0.65")
        score = qwen.cosine(_dummy_rgb(), "bed")
        assert score == pytest.approx(0.65)

    def test_parse_goal_relational(self) -> None:
        response = '{"object": "cup", "relation": "on the kitchen counter", "is_relational": true}'
        qwen = self._make_qwen(response)
        result = qwen.parse_goal("the cup on the kitchen counter")
        assert result["object"] == "cup"
        assert result["is_relational"] is True

    def test_parse_goal_simple(self) -> None:
        response = '{"object": "chair", "relation": "", "is_relational": false}'
        qwen = self._make_qwen(response)
        result = qwen.parse_goal("chair")
        assert result["object"] == "chair"
        assert result["is_relational"] is False

    def test_score_frontier_bad_response_returns_zero(self) -> None:
        qwen = self._make_qwen("I don't know")
        score = qwen.score_frontier(_dummy_rgb(), "toilet")
        assert score == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────
# Tests for RVLFMPolicy logic (no Habitat, no GPU)
# ─────────────────────────────────────────────────────────────

class TestRVLFMPolicyLogic:
    """
    Tests for the core Stage-2 logic in RVLFMPolicy.
    We test the helper methods directly without instantiating the full policy.
    """

    def test_extract_float_edge_cases(self) -> None:
        """Regression: score extraction must never crash or return out-of-range."""
        from vlfm.vlm.qwen_vl import _extract_float
        tricky_inputs = [
            "0.0", "1.0", "0", "1", "", "yes", "no",
            "0.999999", "definitely yes", "0.1 and also 0.9",
            "Score: 0.84.", "NaN", "inf",
        ]
        for text in tricky_inputs:
            val = _extract_float(text)
            assert 0.0 <= val <= 1.0, f"Out of range for input: {repr(text)}"

    def test_stage2_reranking_logic(self) -> None:
        """
        Simulate Stage-2 reranking:
        Given Stage-1 scores and Stage-2 scores, verify the correct
        frontier is selected using R-VLFM's blend formula.
        """
        from vlfm.policy.r_vlfm_policy import STAGE1_WEIGHT, STAGE2_WEIGHT

        # Frontier A: high Stage-1, low Stage-2 (wrong spatial relation)
        s1_a, s2_a = 0.90, 0.10
        # Frontier B: medium Stage-1, high Stage-2 (correct spatial relation)
        s1_b, s2_b = 0.60, 0.95

        combined_a = STAGE1_WEIGHT * s1_a + STAGE2_WEIGHT * s2_a
        combined_b = STAGE1_WEIGHT * s1_b + STAGE2_WEIGHT * s2_b

        # R-VLFM should prefer B despite lower Stage-1 score
        assert combined_b > combined_a, (
            f"R-VLFM should prefer frontier B (relational match) over A. "
            f"combined_a={combined_a:.3f}, combined_b={combined_b:.3f}"
        )

    def test_stage2_weights_sum_to_one(self) -> None:
        from vlfm.policy.r_vlfm_policy import STAGE1_WEIGHT, STAGE2_WEIGHT
        assert STAGE1_WEIGHT + STAGE2_WEIGHT == pytest.approx(1.0), (
            "Stage weights must sum to 1.0 for interpretable scores."
        )

    def test_top_k_verify_positive(self) -> None:
        from vlfm.policy.r_vlfm_policy import TOP_K_VERIFY
        assert TOP_K_VERIFY >= 1, "TOP_K_VERIFY must be at least 1."

    def test_stage1_min_score_in_range(self) -> None:
        from vlfm.policy.r_vlfm_policy import STAGE1_MIN_SCORE
        assert 0.0 <= STAGE1_MIN_SCORE <= 1.0

    def test_parse_and_store_simple_goal(self) -> None:
        """
        _parse_and_store_goal with a simple goal should set is_relational=False.
        """
        from vlfm.policy.r_vlfm_policy import RVLFMPolicy

        # Create a minimal mock policy without calling __init__
        policy = RVLFMPolicy.__new__(RVLFMPolicy)
        policy._object_name = ""
        policy._relational_goal = ""
        policy._is_relational = False

        # Mock QwenVL client parse_goal
        mock_qwen = MagicMock()
        mock_qwen.parse_goal.return_value = {
            "object": "chair",
            "relation": "",
            "is_relational": False,
        }
        policy._qwen = mock_qwen

        policy._parse_and_store_goal("chair")

        assert policy._object_name == "chair"
        assert policy._is_relational is False

    def test_parse_and_store_relational_goal(self) -> None:
        """
        _parse_and_store_goal with a relational goal should set is_relational=True
        and build a well-formed relational_goal string.
        """
        from vlfm.policy.r_vlfm_policy import RVLFMPolicy

        policy = RVLFMPolicy.__new__(RVLFMPolicy)
        policy._object_name = ""
        policy._relational_goal = ""
        policy._is_relational = False

        mock_qwen = MagicMock()
        mock_qwen.parse_goal.return_value = {
            "object": "cup",
            "relation": "on the kitchen counter",
            "is_relational": True,
        }
        policy._qwen = mock_qwen

        policy._parse_and_store_goal("the cup on the kitchen counter")

        assert policy._object_name == "cup"
        assert policy._is_relational is True
        assert "cup" in policy._relational_goal
        assert "kitchen counter" in policy._relational_goal

    def test_parse_and_store_fallback_on_exception(self) -> None:
        """
        If parse_goal throws, _parse_and_store_goal must not crash and
        must fall back to raw text as object_name.
        """
        from vlfm.policy.r_vlfm_policy import RVLFMPolicy

        policy = RVLFMPolicy.__new__(RVLFMPolicy)
        policy._object_name = ""
        policy._relational_goal = ""
        policy._is_relational = False

        mock_qwen = MagicMock()
        mock_qwen.parse_goal.side_effect = RuntimeError("Server unreachable")
        policy._qwen = mock_qwen

        # Should not raise
        policy._parse_and_store_goal("toilet")

        assert policy._object_name == "toilet"
        assert policy._is_relational is False

    def test_get_frontier_rgb_empty_cache(self) -> None:
        """
        _get_frontier_rgb should return None when cache is empty.
        """
        from vlfm.policy.r_vlfm_policy import RVLFMPolicy

        policy = RVLFMPolicy.__new__(RVLFMPolicy)
        policy._frontier_obs_cache = {}

        result = policy._get_frontier_rgb(np.array([1.0, 2.0]))
        assert result is None

    def test_get_frontier_rgb_nearest_key(self) -> None:
        """
        _get_frontier_rgb should return the RGB from the cache key
        nearest to the query frontier.
        """
        from vlfm.policy.r_vlfm_policy import RVLFMPolicy

        policy = RVLFMPolicy.__new__(RVLFMPolicy)

        rgb_near = _dummy_rgb()
        rgb_far = _dummy_rgb()

        # Populate cache: (2, 4) is near [1.0, 2.0], (10, 10) is far
        policy._frontier_obs_cache = {
            (2, 4): (rgb_near, 0.8),
            (10, 10): (rgb_far, 0.3),
        }

        result = policy._get_frontier_rgb(np.array([1.0, 2.0]))
        # Nearest grid key to [1.0/0.5, 2.0/0.5] = [2, 4] is (2, 4)
        assert np.array_equal(result, rgb_near)

    def test_stage2_verify_returns_neutral_on_empty_cache(self) -> None:
        """
        When there are no cached observations, _stage2_verify should
        return a neutral score (0.5) rather than crashing.
        """
        from vlfm.policy.r_vlfm_policy import RVLFMPolicy

        policy = RVLFMPolicy.__new__(RVLFMPolicy)
        policy._frontier_obs_cache = {}
        policy._relational_goal = "a cup on the kitchen counter"
        policy._qwen = MagicMock()

        score = policy._stage2_verify(np.array([1.0, 2.0]))
        assert score == pytest.approx(0.5)
        # Qwen should NOT have been called (no image to verify)
        policy._qwen.verify_relation.assert_not_called()
