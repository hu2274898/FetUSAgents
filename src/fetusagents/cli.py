"""Unified command-line entry point for FetUSAgents.

Run as ``python -m fetusagents --query "..." --input /path/to/image.png``.
The Coordinator routes the request automatically; pass ``--mode`` or
``--task_name`` to override.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .config import load_config
from .coordinator import Coordinator
from .schemas import QueryType, WorkflowResult


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fetusagents",
        description=(
            "Tool-augmented multi-agent system for fetal ultrasound interpretation. "
            "Automatically routes between specific VQA and general (caption / "
            "report / video-summary) workflows."
        ),
    )
    p.add_argument("--query", required=True, help="User question or instruction.")
    p.add_argument(
        "--input",
        default="",
        help="Image file, video file, or directory of frames.",
    )
    p.add_argument(
        "--case_dir",
        default=None,
        help="Optional case directory (overrides --input for the general workflow).",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/run",
        help="Where to write result.json and any intermediate artefacts.",
    )
    p.add_argument(
        "--task_name",
        default=None,
        help="Override the automatically inferred task_type (specific workflow only).",
    )
    p.add_argument(
        "--mode",
        choices=["auto", "specific", "general"],
        default="auto",
        help="Force workflow selection. 'auto' lets the Coordinator decide.",
    )
    p.add_argument("--config", default=None, help="Path to a YAML/JSON config file.")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip heavy model calls and return a deterministic mock result.",
    )
    p.add_argument(
        "--save_report",
        action="store_true",
        help="Also write report.txt and report.md alongside result.json.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client", "autogen_core",
                  "autogen_agentchat", "autogen_ext", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _dispatch(args: argparse.Namespace) -> WorkflowResult:
    cfg = load_config(args.config)
    # IMPORTANT: export env vars BEFORE importing any workflow module.
    # The CV tool wrappers read ``FETALAGENT_*_PYTHON`` at TOOL_CONFIG
    # class-definition time, which happens during ``from ..core import
    # ...``. If we export after the import, the wrappers fall back to
    # bare ``python`` (i.e. the system PATH) and miss the conda envs.
    cfg.export_environment()

    decision = Coordinator().route(
        query=args.query,
        input_path=args.input,
        mode=args.mode,
        task_name=args.task_name,
    )
    logging.info("coordinator decision: %s", json.dumps(decision.to_dict(), ensure_ascii=False))

    if decision.query_type is QueryType.SPECIFIC:
        from .workflows import specific_vqa

        return specific_vqa.run_single(
            image_path=args.input,
            question=args.query,
            task_name=args.task_name,
            cfg=cfg,
            dry_run=args.dry_run,
            decision=decision,
        )

    from .workflows import general

    return general.run_single(
        query=args.query,
        input_path=args.input,
        case_dir=args.case_dir,
        cfg=cfg,
        dry_run=args.dry_run,
        decision=decision,
    )


def _write_outputs(result: WorkflowResult, output_dir: str, save_report: bool) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    if save_report:
        with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write(_format_report_text(result))
        with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(_format_report_markdown(result))
    return result_path


def _format_report_text(result: WorkflowResult) -> str:
    rpt = result.report
    parts = [
        f"Task        : {result.task_type.value}",
        f"Query type  : {result.query_type.value}",
        f"Final answer: {result.final_answer or '-'} ({result.final_option_text or '-'})",
        f"Route       : {result.route or '-'}",
        "",
        "Findings:",
        rpt.findings or "-",
        "",
        "Impression:",
        rpt.impression or "-",
    ]
    if rpt.note:
        parts.extend(["", "Note:", rpt.note])
    return "\n".join(parts)


def _format_report_markdown(result: WorkflowResult) -> str:
    rpt = result.report
    md = [
        f"# Result: {result.task_type.value}",
        "",
        f"- **Query type:** `{result.query_type.value}`",
        f"- **Final answer:** `{result.final_answer or '-'}` — {result.final_option_text or '-'}",
        f"- **Route:** `{result.route or '-'}`",
        f"- **Coordinator confidence:** {result.coordinator.get('confidence', 0.0)}",
        "",
        "## Findings",
        rpt.findings or "_n/a_",
        "",
        "## Impression",
        rpt.impression or "_n/a_",
    ]
    if rpt.note:
        md.extend(["", "## Note", rpt.note])
    return "\n".join(md)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        result = _dispatch(args)
    except Exception as exc:
        logging.exception("workflow failed")
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    result_path = _write_outputs(result, args.output_dir, args.save_report)
    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "result_path": result_path,
                "query_type": result.query_type.value,
                "task_type": result.task_type.value,
                "final_answer": result.final_answer,
                "dry_run": result.dry_run,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
