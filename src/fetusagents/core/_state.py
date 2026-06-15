"""Module-level constants and the global ``TOOL_CONFIG`` singleton.

Everything that other modules need to read at *import* time goes here.
By centralising it we guarantee a single ``TOOL_CONFIG`` instance shared
by every tool runner, regardless of which file imports first.

This module is intentionally tiny and import-cheap: it pulls in nothing
but ``os`` / ``pathlib`` / ``dataclasses``, so any other ``core``
submodule can ``from ._state import TOOL_CONFIG`` without circular risk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# _SCRIPT_DIR points at the package directory ``src/fetusagents/``, which
# is where the ``tools/`` and ``external_tools/`` subtrees live.
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
# Checkpoints are NOT shipped with the repository. By default we look
# for a sibling ``FetalAgent_ckpt/`` next to the repo root; override via
# the ``FETALAGENT_CKPT_DIR`` environment variable.
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_CKPT_DIR = Path(
    os.environ.get("FETALAGENT_CKPT_DIR", str(_REPO_ROOT.parent / "FetalAgent_ckpt"))
).resolve()


@dataclass
class ToolConfig:
    """Configuration for external tool paths.

    All paths default to locations relative to ``src/fetusagents/`` (the
    package directory). Override via environment variables or by mutating
    this dataclass.
    Conda environment Python paths must be set to match your installation.
    """

    # Conda environments -- set these to your own conda env paths
    hxt_base_python: str = os.environ.get("FETALAGENT_HXT_BASE_PYTHON", "python")
    fetalclip_python: str = os.environ.get("FETALAGENT_FETALCLIP_PYTHON", "python")
    fetalclip2_python: str = os.environ.get("FETALAGENT_FETALCLIP2_PYTHON", "python")
    experiment_aaai_python: str = os.environ.get("FETALAGENT_EXPERIMENT_AAAI_PYTHON", "python")
    usfm_python: str = os.environ.get("FETALAGENT_USFM_PYTHON", "python")

    # Tool directories (relative to project root)
    aop_sam_dir: str = str(_SCRIPT_DIR / "external_tools" / "AoP_SAM")
    usfm_aop_dir: str = str(_SCRIPT_DIR / "external_tools" / "USFM_aop")
    csm_hc_dir: str = str(_SCRIPT_DIR / "external_tools" / "CSM_hc")
    usfm_hc_dir: str = str(_SCRIPT_DIR / "external_tools" / "USFM_hc")
    ga_algo1_dir: str = str(_SCRIPT_DIR / "external_tools" / "ga_radimagenet")
    ga_algo2_dir: str = str(_SCRIPT_DIR / "external_tools" / "ga_fetalclip")
    ga_algo3_dir: str = str(_SCRIPT_DIR / "external_tools" / "ga_convnext")
    keyframe_cls6_dir: str = str(_SCRIPT_DIR / "external_tools" / "keyframe_cls6")
    plane_fetalclip_dir: str = str(_SCRIPT_DIR / "external_tools" / "plane_fetalclip")
    plane_fulora_dir: str = str(_SCRIPT_DIR / "external_tools" / "plane_fulora")
    brain_subplane_fetalclip_dir: str = str(_SCRIPT_DIR / "external_tools" / "brain_subplane_fetalclip")
    agent_tools_dir: str = str(_SCRIPT_DIR / "cli_wrappers")

    # Checkpoints -- expected in sibling folder FetalAgent_ckpt/
    aop_sam_ckpt: str = str(_CKPT_DIR / "aop_sam_fold0.pth")
    upernet_aop_ckpt: str = str(_CKPT_DIR / "upernet_aop_fold0.pth")
    brain_subplane_fetalclip_ckpt: str = str(_CKPT_DIR / "brain_subplane_fetalclip.ckpt")
    brain_subplane_resnet_ckpt: str = str(_CKPT_DIR / "brain_subplane_resnet.pth")
    brain_subplane_vit_ckpt: str = str(_CKPT_DIR / "brain_subplane_vit.pth")
    stomach_fetalclip_ckpt: str = str(_CKPT_DIR / "stomach_fetalclip.ckpt")
    stomach_samus_ckpt: str = str(_CKPT_DIR / "stomach_samus.pth")
    abdomen_fetalclip_ckpt: str = str(_CKPT_DIR / "abdomen_fetalclip.ckpt")
    abdomen_samus_ckpt: str = str(_CKPT_DIR / "abdomen_samus.pth")
    samus_base_ckpt: str = str(_CKPT_DIR / "SAMUS.pth")
    nnunet_predict: str = os.environ.get("FETALAGENT_NNUNET_PREDICT", "nnUNetv2_predict")
    keyframe_cls6_config: str = str(_SCRIPT_DIR / "external_tools" / "keyframe_cls6" / "config" / "classification.yml")

    # Timeouts
    default_timeout: int = 1800


# Global config instance (singleton). All tool runners reference this.
TOOL_CONFIG = ToolConfig()


# Misc module-level regex used across files
_FNAME_EXT_RE = r"(?:png|jpg|jpeg|bmp|tif|tiff|webp)"
