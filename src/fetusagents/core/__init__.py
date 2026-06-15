"""``fetus_core`` — shared infrastructure for both workflows.

Layout::

    _state.py        TOOL_CONFIG, _CKPT_DIR, _SCRIPT_DIR, _FNAME_EXT_RE
    llm.py           build_model_client, ToolResult, run_tool_subprocess
    biometry.py      pixel CSV, HC/AC/GA helpers, percentile reference tables
    image_utils.py   _safe_load_pil / _make_overlay / _mask_* / _dice / vote helpers
    parsing.py       _parse_filename_* / _normalize_video_plane_label / etc.
    tool_runners.py  17 run_*_tool subprocess wrappers + _weighted_vote_ensemble_ga
    agent_base.py    ToolAgent base type shared by both workflows
"""
from __future__ import annotations

# State / config
from ._state import (
    _CKPT_DIR,
    _FNAME_EXT_RE,
    _SCRIPT_DIR,
    TOOL_CONFIG,
    ToolConfig,
)

# LLM + subprocess infrastructure
from .llm import (
    ToolResult,
    build_model_client,
    run_tool_subprocess,
)

# Biometry helpers
from .biometry import (
    GA_TO_HC_PARAMS,
    _ac_mm_from_mask_array,
    _ellipse_circumference_mm_from_mask_array,
    _extract_lmp_ga_weeks,
    _fmt_percentile,
    _format_ga_weeks_days,
    _ga_label_to_weeks,
    _hadlock_ga_weeks_from_ac_mm,
    _hc_mm_from_mask_array,
    _hc_percentile_sanity_check,
    _largest_component_edge,
    _load_ga_reference_table,
    _nearest_ga_row,
    _parse_expert_per_image,
    _percentile_assessment,
    _plane_display_name,
    _round_1dp,
    ensure_pixel_csv,
    float_weeks_to_weeks_days,
    hc_from_ga_weeks,
    hc_range_from_ga_weeks,
    parse_pixel_size_csv,
    weeks_days_to_float_weeks,
)

# Image / mask utilities
from .image_utils import (
    _agent_outputs_dir,
    _apply_postprocess,
    _compute_ellipse_residual,
    _concat_side_by_side,
    _dice_masks,
    _keep_largest_component,
    _load_mask_binary_cv2,
    _majority_voting,
    _make_overlay,
    _make_single_image_case_dir,
    _make_square_pil,
    _mask_to_raw_array,
    _pil_to_agimage,
    _safe_load_pil,
)

# Parsing helpers
from .parsing import (
    _is_video_summary_request,
    _normalize_video_plane_label,
    _parse_filename_colon_text,
    _parse_filename_colon_value,
    _parse_filename_label_probs,
    _parse_seg_judge_output,
    _resolve_image_key,
)

# Tool runners
from .tool_runners import (
    _weighted_vote_ensemble_ga,
    run_abdomen_fetalclip_samus_seg_tool,
    run_abdomen_fetalclip_seg_tool,
    run_aop_sam_tool,
    run_brain_subplane_fetalclip_tool,
    run_brain_subplane_resnet_tool,
    run_brain_subplane_vit_tool,
    run_csm_hc_tool,
    run_ga_algo1_tool,
    run_ga_algo2_tool,
    run_ga_algo3_tool,
    run_nnunet_hc_tool,
    run_plane_fetalclip_tool,
    run_plane_fulora_tool,
    run_stomach_fetalclip_samus_seg_tool,
    run_stomach_fetalclip_seg_tool,
    run_stomach_nnunet_seg_tool,
    run_upernet_aop_tool,
    run_usfm_aop_tool,
    run_usfm_hc_tool,
    run_video_keyframe_tool,
)

# Orchestrator + agents moved to ``fetusagents.general`` — see
# :mod:`fetusagents.general.pipeline` and :mod:`fetusagents.general.experts`.


# Explicit __all__ so ``from fetus_core import *`` also re-exports
# names that start with an underscore (the original main.py exposed many
# such helpers as part of its de-facto public API; we preserve that).
__all__ = [
    # _state
    "_CKPT_DIR", "_FNAME_EXT_RE", "_SCRIPT_DIR", "TOOL_CONFIG", "ToolConfig",
    # llm
    "ToolResult", "build_model_client", "run_tool_subprocess",
    # biometry
    "GA_TO_HC_PARAMS",
    "_ac_mm_from_mask_array", "_ellipse_circumference_mm_from_mask_array",
    "_extract_lmp_ga_weeks", "_fmt_percentile", "_format_ga_weeks_days",
    "_ga_label_to_weeks", "_hadlock_ga_weeks_from_ac_mm", "_hc_mm_from_mask_array",
    "_hc_percentile_sanity_check", "_largest_component_edge",
    "_load_ga_reference_table", "_nearest_ga_row", "_parse_expert_per_image",
    "_percentile_assessment", "_plane_display_name", "_round_1dp",
    "ensure_pixel_csv", "float_weeks_to_weeks_days",
    "hc_from_ga_weeks", "hc_range_from_ga_weeks", "parse_pixel_size_csv",
    "weeks_days_to_float_weeks",
    # image_utils
    "_agent_outputs_dir", "_apply_postprocess", "_compute_ellipse_residual",
    "_concat_side_by_side", "_dice_masks", "_keep_largest_component",
    "_load_mask_binary_cv2", "_majority_voting", "_make_overlay",
    "_make_single_image_case_dir", "_make_square_pil", "_mask_to_raw_array",
    "_pil_to_agimage", "_safe_load_pil",
    # parsing
    "_is_video_summary_request", "_normalize_video_plane_label",
    "_parse_filename_colon_text", "_parse_filename_colon_value",
    "_parse_filename_label_probs", "_parse_seg_judge_output", "_resolve_image_key",
    # tool_runners
    "_weighted_vote_ensemble_ga",
    "run_abdomen_fetalclip_samus_seg_tool", "run_abdomen_fetalclip_seg_tool",
    "run_aop_sam_tool", "run_brain_subplane_fetalclip_tool",
    "run_brain_subplane_resnet_tool", "run_brain_subplane_vit_tool",
    "run_csm_hc_tool", "run_ga_algo1_tool", "run_ga_algo2_tool", "run_ga_algo3_tool",
    "run_nnunet_hc_tool", "run_plane_fetalclip_tool", "run_plane_fulora_tool",
    "run_stomach_fetalclip_samus_seg_tool", "run_stomach_fetalclip_seg_tool",
    "run_stomach_nnunet_seg_tool", "run_upernet_aop_tool", "run_usfm_aop_tool",
    "run_usfm_hc_tool", "run_video_keyframe_tool",
]
