"""Model client, ToolResult schema, and subprocess runner.

This file collects the *infrastructure* layer of the orchestrator:
nothing in here understands clinical concepts; the next layer
(``biometry`` / ``tool_runners``) does. Importing this module is cheap
and has no side effects beyond initialising autogen clients (which
themselves are lazy and just store config until used).
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autogen_ext.models.openai import OpenAIChatCompletionClient


# Model client configuration
def build_model_client() -> OpenAIChatCompletionClient:
    model_name = os.environ.get("OPENAI_MODEL", "gpt-5.1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAIChatCompletionClient(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
        },
    )


# Tool Result Schema
@dataclass
class ToolResult:
    """Standardized result from tool execution."""
    tool_name: str
    ok: bool
    per_image: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None
    logs: Dict[str, str] = field(default_factory=dict)
    artifacts_dir: Optional[str] = None


# Subprocess Runner
def run_tool_subprocess(
    python_path: str,
    script_path: str,
    args: List[str],
    cwd: Optional[str] = None,
    timeout: int = 1800,
    env_extra: Optional[Dict[str, str]] = None,
    log_prefix: Optional[str] = None,
    print_regexes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a tool script via subprocess, streaming output live to the console.
    Returns dict with returncode, combined stdout, and cmd.
    """
    import selectors

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    # Force unbuffered output so users can see progress while tools run
    env.setdefault("PYTHONUNBUFFERED", "1")

    prefix = log_prefix or os.path.basename(script_path)
    print_tool_output = os.environ.get("AGENT_PRINT_TOOL_OUTPUT", "1") not in ("0", "false", "False")
    print_tool_cmd = os.environ.get("AGENT_PRINT_TOOL_CMD", "0") in ("1", "true", "True", "yes", "Yes")
    heartbeat_sec = float(os.environ.get("AGENT_TOOL_HEARTBEAT_SEC", "60"))
    compiled_regexes: List[re.Pattern[str]] = []
    if print_regexes:
        compiled_regexes = [re.compile(p) for p in print_regexes]

    # -u: unbuffered
    cmd = [python_path, "-u", script_path] + args

    try:
        if print_tool_output and print_tool_cmd:
            print(f">>> [Tool:{prefix}] CMD: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,  # nosec - controlled internally
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)

        start = time.time()
        last_output = start
        out_chunks: List[str] = []

        while True:
            now = time.time()
            if timeout and (now - start) > timeout:
                try:
                    proc.kill()
                except Exception:
                    pass
                out_chunks.append(f"\n[{prefix}] ERROR: Timeout after {timeout}s\n")
                break

            events = sel.select(timeout=0.5)
            if events:
                for key, _mask in events:
                    line = key.fileobj.readline()
                    if not line:
                        # EOF
                        try:
                            sel.unregister(key.fileobj)
                        except Exception:
                            pass
                        break
                    out_chunks.append(line)
                    last_output = now
                    if print_tool_output:
                        # Print only result lines (and always print obvious errors)
                        is_errorish = ("Traceback" in line) or ("ERROR" in line) or ("Error" in line) or ("Exception" in line)
                        is_result = True
                        if compiled_regexes:
                            is_result = any(rx.search(line) for rx in compiled_regexes)
                        if is_errorish or is_result:
                            print(f"[{prefix}] {line}", end="", flush=True)
            else:
                # Heartbeat so users know we're still running even if the tool is silent
                if print_tool_output and heartbeat_sec > 0 and (now - last_output) >= heartbeat_sec:
                    print(f"[{prefix}] ...running...", flush=True)
                    last_output = now

            if proc.poll() is not None:
                # Drain remaining output
                try:
                    rest = proc.stdout.read()
                except Exception:
                    rest = ""
                if rest:
                    out_chunks.append(rest)
                    if print_tool_output:
                        for l in rest.splitlines(True):
                            is_errorish = ("Traceback" in l) or ("ERROR" in l) or ("Error" in l) or ("Exception" in l)
                            is_result = True
                            if compiled_regexes:
                                is_result = any(rx.search(l) for rx in compiled_regexes)
                            if is_errorish or is_result:
                                print(f"[{prefix}] {l}", end="", flush=True)
                break

        try:
            sel.close()
        except Exception:
            pass

        returncode = proc.returncode if proc.returncode is not None else -1
        stdout_text = "".join(out_chunks)
        return {
            "ok": returncode == 0,
            "returncode": returncode,
            "stdout": stdout_text,
            "stderr": "",  # merged into stdout
            "cmd": cmd,
        }

    except Exception as e:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "cmd": cmd,
            "error": str(e),
        }
