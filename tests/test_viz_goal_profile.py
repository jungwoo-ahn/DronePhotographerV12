"""Tests for the goal-profile inspector's Python side.

The page's JS re-implements the vocabulary lookup and the prompt serializer, so the thing worth
protecting is that it is DRIVEN by the repo's tables rather than carrying its own copies — the
failure mode `docs/v4_session_changes.md` section 5 documents. These check that the exported
config is the vocabulary verbatim, that an authored preset is internally consistent under the
visible-bbox convention, and that the built page is self-contained.

Pure/deterministic — no data, no network, no browser.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from src.common.goal_space import DEFAULT_GOAL_KEYS, RENDER_HEIGHT, RENDER_WIDTH
from src.data.lerobot_export import crop_phrase
from src.goal_authoring import vocab

# The script lives in scripts/ and is not a package, so import it by path.
_SPEC = importlib.util.spec_from_file_location(
    "viz_goal_profile", Path(__file__).resolve().parents[1] / "scripts" / "viz_goal_profile.py"
)
viz = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(viz)


def test_exported_tables_are_the_vocab_tables():
    cfg = viz.page_config()
    for name, table in (("SHOT_SIZE", vocab.SHOT_SIZE), ("BODY_FRAMING", vocab.BODY_FRAMING),
                        ("ELEVATION", vocab.ELEVATION), ("PLACE_X", vocab.PLACE_X),
                        ("PLACE_Y", vocab.PLACE_Y)):
        assert cfg["tables"][name] == [[k, lo, hi, c] for k, (lo, hi, c) in table.items()], name


def test_exported_table_order_is_preserved():
    # `vocab._classify` returns the FIRST band containing the value, so order is semantics.
    cfg = viz.page_config()
    assert [row[0] for row in cfg["tables"]["PLACE_X"]] == list(vocab.PLACE_X)


def test_exported_config_carries_the_real_intrinsics_and_keys():
    cfg = viz.page_config()
    assert cfg["goal_keys"] == list(DEFAULT_GOAL_KEYS)
    assert cfg["render_w"] == RENDER_WIDTH and cfg["render_h"] == RENDER_HEIGHT
    # 24 mm on a 12.8 x 9.6 mm sensor -> square pixels at this resolution
    assert cfg["fx"] == pytest.approx(1920.0) and cfg["fy"] == pytest.approx(1920.0)


@pytest.mark.parametrize("preset", viz.preset_goals())
def test_preset_goals_are_consistent_with_the_visible_bbox_convention(preset):
    g, crop = preset["goal"], preset["crop"]
    assert set(g) == set(DEFAULT_GOAL_KEYS)
    # the visible box must lie inside the frame — that is what "visible" means
    assert g["object_center_y"] - g["bbox_y_offset"] >= -1e-6
    assert g["object_center_y"] + g["bbox_y_offset"] <= RENDER_HEIGHT + 1e-6
    assert g["object_center_x"] - g["bbox_x_offset"] >= -1e-6
    assert g["object_center_x"] + g["bbox_x_offset"] <= RENDER_WIDTH + 1e-6
    # and the crop fractions must account for whatever was cut
    assert 0.0 <= crop["top"] < 1.0 and 0.0 <= crop["bot"] < 1.0
    assert crop["top"] + crop["bot"] < 1.0


def test_fit_authored_box_records_the_cut_it_makes():
    # a subject twice the frame's height, centred: half cut off each end
    g = {k: 0.0 for k in DEFAULT_GOAL_KEYS}
    g["object_center_x"], g["object_center_y"] = RENDER_WIDTH / 2, RENDER_HEIGHT / 2
    fitted, crop = viz._fit_authored_box(g, 100.0, RENDER_HEIGHT)
    assert crop["top"] == pytest.approx(0.25) and crop["bot"] == pytest.approx(0.25)
    assert fitted["bbox_y_offset"] == pytest.approx(RENDER_HEIGHT / 2)
    assert crop_phrase(crop["top"], crop["bot"]) == "cropped at both the head and the feet"


def test_fit_authored_box_leaves_a_fitting_subject_alone():
    g = {k: 0.0 for k in DEFAULT_GOAL_KEYS}
    g["object_center_x"], g["object_center_y"] = RENDER_WIDTH / 2, RENDER_HEIGHT / 2
    fitted, crop = viz._fit_authored_box(g, 100.0, 200.0)
    assert crop == {"top": 0.0, "bot": 0.0}
    assert fitted["bbox_y_offset"] == pytest.approx(200.0)
    assert fitted["object_center_y"] == pytest.approx(RENDER_HEIGHT / 2)


def test_built_page_is_self_contained_and_carries_a_parsable_payload():
    html = viz.build_html(viz.page_config(), [], {"goal": viz.preset_goals()[0]["goal"],
                                                  "crop": viz.preset_goals()[0]["crop"],
                                                  "presets": viz.preset_goals(), "opts": {}})
    assert "/*__PAYLOAD__*/" not in html
    # no external fetches: the page has to open from a file:// path with no network
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)
    payload = json.loads(re.search(r"const DATA = (\{.*?\});\nconst C", html, re.S).group(1))
    assert payload["cfg"]["goal_keys"] == list(DEFAULT_GOAL_KEYS)
    assert len(payload["initial"]["presets"]) == len(viz.PRESET_CATEGORIES)
