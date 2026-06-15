"""Path / environment configuration for FetUSAgents.

The CV tools that ship under :mod:`fetusagents.core` read a handful of
environment variables (``OPENAI_API_KEY``, ``FETALAGENT_CKPT_DIR``, a
per-conda-env Python path each, ...). Rather than scatter ``os.getenv``
calls across the codebase, this module loads a YAML config once and
exports the resolved values via :meth:`FetUSConfig.export_environment`.

Concrete defaults live in ``configs/default.yaml`` and
``configs/paths.example.yaml``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _try_yaml_load(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return _minimal_yaml_load(text)
    return yaml.safe_load(text) or {}


def _minimal_yaml_load(text: str) -> Dict[str, Any]:
    """Tiny YAML-subset parser for ``key: value`` style configs.

    Pulled in so the package can read its own example configs even when
    PyYAML is not installed in the environment running the routing tests.
    Only flat ``key: value`` and one-level nesting are supported.
    """

    out: Dict[str, Any] = {}
    stack = [(0, out)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: Dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent + 2, child))
        else:
            v = value.strip().strip('"').strip("'")
            if v.lower() in ("true", "false"):
                parent[key.strip()] = v.lower() == "true"
            else:
                try:
                    parent[key.strip()] = int(v)
                except ValueError:
                    try:
                        parent[key.strip()] = float(v)
                    except ValueError:
                        parent[key.strip()] = v
    return out


@dataclass
class FetUSConfig:
    """Resolved configuration. Always created via :func:`load_config`."""

    # Checkpoint root and RAG database root.
    fetalagent_ckpt_dir: Optional[str] = None
    rag_db_path: Optional[str] = None

    # Conda / virtualenv Python executables used by the CV tool wrappers.
    # Each subprocess tool calls one of these to pick up its trained model.
    hxt_base_python: Optional[str] = None
    fetalclip_python: Optional[str] = None
    fetalclip2_python: Optional[str] = None
    experiment_aaai_python: Optional[str] = None
    usfm_python: Optional[str] = None
    nnunet_predict: Optional[str] = None

    # LLM client config.
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5.1"
    openai_base_url: str = "https://api.openai.com/v1"

    # Behaviour knobs.
    tool_timeout: int = 1800
    output_dir: str = "outputs"
    log_level: str = "INFO"

    # Bag for any extra fields read from YAML.
    extras: Dict[str, Any] = field(default_factory=dict)

    def export_environment(self) -> None:
        """Push selected fields into ``os.environ``.

        The CV tool wrappers read several variables directly (e.g.
        ``FETALAGENT_HXT_BASE_PYTHON``, ``OPENAI_API_KEY``), so we set
        them here once before any workflow runs.
        """
        env_map = {
            "FETALAGENT_CKPT_DIR": self.fetalagent_ckpt_dir,
            "FETALAGENT_HXT_BASE_PYTHON": self.hxt_base_python,
            "FETALAGENT_FETALCLIP_PYTHON": self.fetalclip_python,
            "FETALAGENT_FETALCLIP2_PYTHON": self.fetalclip2_python,
            "FETALAGENT_EXPERIMENT_AAAI_PYTHON": self.experiment_aaai_python,
            "FETALAGENT_USFM_PYTHON": self.usfm_python,
            "FETALAGENT_NNUNET_PREDICT": self.nnunet_predict,
            "OPENAI_API_KEY": self.openai_api_key,
            "OPENAI_MODEL": self.openai_model,
            "OPENAI_BASE_URL": self.openai_base_url,
        }
        for k, v in env_map.items():
            if v is not None and v != "":
                os.environ.setdefault(k, str(v))


def _read_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        return _try_yaml_load(text)
    return json.loads(text)


def _candidate_config_paths() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("FETUSAGENTS_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_REPO_ROOT / "configs" / "paths.local.yaml")
    candidates.append(_REPO_ROOT / "configs" / "paths.example.yaml")
    candidates.append(_REPO_ROOT / "configs" / "default.yaml")
    return candidates


def load_config(path: Optional[str] = None) -> FetUSConfig:
    """Resolve configuration from explicit path, env var, or built-in defaults.

    Resolution order:
      1. ``path`` argument when given;
      2. ``$FETUSAGENTS_CONFIG`` environment variable;
      3. ``configs/paths.local.yaml`` (gitignored, user-specific);
      4. ``configs/paths.example.yaml`` then ``configs/default.yaml``.

    Environment variables override file contents for the OpenAI fields.
    """
    sources: list[Dict[str, Any]] = []
    if path:
        sources.append(_read_yaml_or_json(Path(path)))
    else:
        for candidate in _candidate_config_paths():
            if candidate.is_file():
                sources.append(_read_yaml_or_json(candidate))
                break

    merged: Dict[str, Any] = {}
    for s in sources:
        merged.update(s.get("paths", {}) if isinstance(s.get("paths"), dict) else {})
        merged.update(s.get("python_envs", {}) if isinstance(s.get("python_envs"), dict) else {})
        merged.update(s.get("llm", {}) if isinstance(s.get("llm"), dict) else {})
        for k in ("tool_timeout", "output_dir", "log_level"):
            if k in s:
                merged[k] = s[k]

    cfg = FetUSConfig(
        fetalagent_ckpt_dir=merged.get("fetalagent_ckpt_dir"),
        rag_db_path=merged.get("rag_db_path"),
        hxt_base_python=merged.get("hxt_base_python"),
        fetalclip_python=merged.get("fetalclip_python"),
        fetalclip2_python=merged.get("fetalclip2_python"),
        experiment_aaai_python=merged.get("experiment_aaai_python"),
        usfm_python=merged.get("usfm_python"),
        nnunet_predict=merged.get("nnunet_predict"),
        openai_api_key=os.environ.get("OPENAI_API_KEY") or merged.get("openai_api_key"),
        openai_model=os.environ.get("OPENAI_MODEL") or merged.get("openai_model", "gpt-5-mini"),
        openai_base_url=os.environ.get("OPENAI_BASE_URL")
        or merged.get("openai_base_url", "https://api.openai.com/v1"),
        tool_timeout=int(merged.get("tool_timeout", 1800)),
        output_dir=merged.get("output_dir", "outputs"),
        log_level=merged.get("log_level", "INFO"),
        extras=merged,
    )
    return cfg
