from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    output_dir: str
    discovered_files: List[str]
    metrics: Dict[str, float]


def _iter_files(root: str) -> List[str]:
    p = Path(root)
    if not p.exists():
        return []
    files: List[str] = []
    for fp in p.rglob("*"):
        if fp.is_file():
            files.append(str(fp))
    return sorted(files)


def _parse_metrics_from_text(text: str) -> Dict[str, float]:
    # Heuristics: capture common metric keys from logs / evaluate.txt / json dumps
    patterns = [
        r"\bR@1\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bR@5\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bR@10\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bRecall@1\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bRecall@5\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bRecall@10\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bacc(?:uracy)?\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bCIDEr\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bSPICE\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bMETEOR\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bROUGE_L\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bBLEU[-_ ]?1\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bBLEU[-_ ]?2\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bBLEU[-_ ]?3\b[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bBLEU[-_ ]?4\b[:=]\s*([0-9]*\.?[0-9]+)",
    ]
    keys = [
        "R@1",
        "R@5",
        "R@10",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "acc",
        "CIDEr",
        "SPICE",
        "METEOR",
        "ROUGE_L",
        "BLEU-1",
        "BLEU-2",
        "BLEU-3",
        "BLEU-4",
    ]

    out: Dict[str, float] = {}
    for k, pat in zip(keys, patterns):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            out[k] = float(m.group(1))
        except Exception:
            pass
    return out


def _try_load_json_metrics(files: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for f in files:
        if not f.lower().endswith(".json"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception:
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (int, float)) and k not in out:
                    out[k] = float(v)
    return out


def run_python_script(
    script_path: str,
    args: List[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    output_dir_hint: Optional[str] = None,
    timeout_s: Optional[int] = None,
) -> RunResult:
    cmd = [sys.executable, script_path, *args]
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )

    out_dir = output_dir_hint or (cwd or os.getcwd())
    discovered = _iter_files(out_dir)
    merged_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    metrics = _parse_metrics_from_text(merged_text)
    metrics.update({k: v for k, v in _try_load_json_metrics(discovered).items() if k not in metrics})

    return RunResult(
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        output_dir=out_dir,
        discovered_files=discovered,
        metrics=metrics,
    )

