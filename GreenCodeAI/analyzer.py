"""
analyzer.py — Read a Python training script and detect common inefficient patterns.

Uses AST when possible for reliable detection; falls back to regex for edge cases.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnalysisIssue:
    """One detected issue in the training script."""

    code: str
    title: str
    detail: str
    severity: str = "medium"  # low | medium | high


@dataclass
class AnalysisResult:
    """Full result of analyzing a single Python file."""

    file_path: Path
    issues: list[AnalysisIssue] = field(default_factory=list)
    raw_text: str = ""


def _read_file_text(path: Path) -> str:
    """
    Read the entire file as UTF-8 text.

    Raises:
        FileNotFoundError: If the path does not exist.
        OSError: If the file cannot be read.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_parse_ast(source: str) -> ast.AST | None:
    """
    Parse Python source into an AST.

    Returns:
        ast.AST on success, or None if the source is not valid Python.
    """
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _literal_int_value(node: ast.AST | None) -> int | None:
    """
    Extract an integer literal from an AST node if possible.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Num):  # pragma: no cover — Python < 3.8 compatibility
        if isinstance(node.n, int):
            return int(node.n)
    return None


def _name_or_attr_str(node: ast.AST | None) -> str | None:
    """
    Best-effort string for a Name or Attribute chain (e.g. 'torch.float16').
    """
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_or_attr_str(node.value)
        if base:
            return f"{base}.{node.attr}"
        return node.attr
    return None


class _TrainingPatternVisitor(ast.NodeVisitor):
    """
    Walk the AST and collect signals used for green-efficiency checks.
    """

    def __init__(self) -> None:
        """Create empty counters for the training-pattern visitor."""
        self.batch_sizes: list[int] = []
        self.num_workers_values: list[int] = []
        self.data_loader_calls: int = 0
        self.mixed_precision_flags: list[bool] = []
        self.full_finetune_hints: list[str] = []
        self.optimizer_on_all_params: bool = False

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: ANN401 — ast visitor API
        """Find DataLoader settings, autocast usage, and full-parameter optimizers."""
        func_name = _name_or_attr_str(node.func)
        if func_name and func_name.endswith("DataLoader"):
            self.data_loader_calls += 1
            for kw in node.keywords:
                if kw.arg == "batch_size":
                    v = _literal_int_value(kw.value)
                    if v is not None:
                        self.batch_sizes.append(v)
                if kw.arg == "num_workers":
                    v = _literal_int_value(kw.value)
                    if v is not None:
                        self.num_workers_values.append(v)

        if func_name and func_name.endswith("GradScaler"):
            # Having a GradScaler is a hint mixed precision may be used.
            pass

        if func_name and (func_name.endswith("autocast") or func_name == "autocast"):
            for kw in node.keywords:
                if kw.arg == "enabled":
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                        self.mixed_precision_flags.append(bool(kw.value.value))

        if func_name and (func_name.endswith("Adam") or "AdamW" in func_name):
            if node.args:
                first = node.args[0]
                if isinstance(first, ast.Attribute) and first.attr == "parameters":
                    if isinstance(first.value, ast.Name) and first.value.id in {"model", "net"}:
                        self.optimizer_on_all_params = True

        self.generic_visit(node)
        return None

    def visit_Assign(self, node: ast.Assign) -> Any:  # noqa: ANN401
        """Pick up batch_size, num_workers, mixed_precision, and fine-tune flags from assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id.lower()
                if name == "batch_size":
                    v = _literal_int_value(node.value)
                    if v is not None:
                        self.batch_sizes.append(v)
                if name == "num_workers":
                    v = _literal_int_value(node.value)
                    if v is not None:
                        self.num_workers_values.append(v)
                if "mixed" in name and "precision" in name:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                        self.mixed_precision_flags.append(bool(node.value.value))
                if name in {"train_full_model", "full_finetune", "finetune_full"}:
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                        if node.value.value:
                            self.full_finetune_hints.append(name)

        self.generic_visit(node)
        return None


def _regex_fallback_signals(text: str) -> dict[str, Any]:
    """
    When AST parsing fails, use simple regex heuristics on the raw text.

    Returns:
        A dict of coarse signals for downstream issue generation.
    """
    signals: dict[str, Any] = {
        "batch_sizes": [],
        "num_workers": [],
        "mixed_precision_false": False,
        "full_finetune": False,
    }
    for m in re.finditer(r"batch_size\s*=\s*(\d+)", text, flags=re.IGNORECASE):
        signals["batch_sizes"].append(int(m.group(1)))
    for m in re.finditer(r"num_workers\s*=\s*(\d+)", text, flags=re.IGNORECASE):
        signals["num_workers"].append(int(m.group(1)))
    if re.search(r"mixed_precision\s*=\s*False", text, flags=re.IGNORECASE):
        signals["mixed_precision_false"] = True
    if re.search(r"train_full_model\s*=\s*True", text, flags=re.IGNORECASE):
        signals["full_finetune"] = True
    return signals


def analyze_training_script(file_path: str | Path) -> AnalysisResult:
    """
    Analyze a Python training script for inefficient patterns.

    Args:
        file_path: Path to the .py file.

    Returns:
        AnalysisResult containing a list of issues and the file's text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not a file.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    text = _read_file_text(path)
    issues: list[AnalysisIssue] = []

    tree = _safe_parse_ast(text)
    if tree is None:
        issues.append(
            AnalysisIssue(
                code="SYNTAX",
                title="Invalid Python syntax",
                detail="The file could not be parsed as Python. Regex-only checks will be used.",
                severity="high",
            )
        )
        sig = _regex_fallback_signals(text)
        for bs in sig["batch_sizes"]:
            if bs > 128:
                issues.append(
                    AnalysisIssue(
                        code="LARGE_BATCH",
                        title="Large batch size",
                        detail=f"Detected batch_size = {bs} (> 128). Large batches increase memory and energy spikes.",
                        severity="medium",
                    )
                )
        for nw in sig["num_workers"]:
            if nw == 0:
                issues.append(
                    AnalysisIssue(
                        code="WORKERS_ZERO",
                        title="DataLoader num_workers is 0",
                        detail="Training often waits on the CPU to load data. Try 2–8 workers.",
                        severity="medium",
                    )
                )
        if sig["mixed_precision_false"]:
            issues.append(
                AnalysisIssue(
                    code="NO_AMP",
                    title="Mixed precision disabled",
                    detail="FP16/BF16 mixed precision can cut GPU energy for many models.",
                    severity="medium",
                )
            )
        if sig["full_finetune"]:
            issues.append(
                AnalysisIssue(
                    code="FULL_FT",
                    title="Full fine-tuning detected",
                    detail="Updating all parameters increases compute. Parameter-efficient methods help.",
                    severity="high",
                )
            )
        if not sig["num_workers"] and "DataLoader" in text and "num_workers" not in text:
            issues.append(
                AnalysisIssue(
                    code="LOADER_DEFAULT",
                    title="Inefficient DataLoader usage",
                    detail="DataLoader without tuned num_workers/pin_memory can bottleneck training.",
                    severity="low",
                )
            )
        return AnalysisResult(file_path=path, issues=issues, raw_text=text)

    visitor = _TrainingPatternVisitor()
    visitor.visit(tree)

    if visitor.batch_sizes and max(visitor.batch_sizes) > 128:
        issues.append(
            AnalysisIssue(
                code="LARGE_BATCH",
                title="Large batch size",
                detail=f"Detected batch_size up to {max(visitor.batch_sizes)} (> 128).",
                severity="medium",
            )
        )

    if visitor.num_workers_values and min(visitor.num_workers_values) == 0:
        issues.append(
            AnalysisIssue(
                code="WORKERS_ZERO",
                title="DataLoader num_workers is 0",
                detail="Use multiple workers so the GPU spends less time idle waiting for batches.",
                severity="medium",
            )
        )

    has_autocast_enabled_true = any(visitor.mixed_precision_flags)
    mixed_precision_literal_false = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and "mixed" in t.id.lower() and "precision" in t.id.lower() for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value is False
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
    )

    if mixed_precision_literal_false or (not has_autocast_enabled_true and "autocast" not in text):
        # If script explicitly disables mixed precision, or never mentions autocast/amp.
        if "torch.cuda.amp" in text or "autocast" in text:
            if mixed_precision_literal_false:
                issues.append(
                    AnalysisIssue(
                        code="NO_AMP",
                        title="Mixed precision disabled",
                        detail="AMP modules are imported or referenced, but mixed precision is set to False.",
                        severity="medium",
                    )
                )
        else:
            issues.append(
                AnalysisIssue(
                    code="NO_AMP",
                    title="Mixed precision likely missing",
                    detail="No clear torch.cuda.amp / autocast usage detected.",
                    severity="medium",
                )
            )

    if visitor.full_finetune_hints or visitor.optimizer_on_all_params:
        issues.append(
            AnalysisIssue(
                code="FULL_FT",
                title="Full fine-tuning detected",
                detail="Training all model parameters is energy-heavy. Consider LoRA / adapters.",
                severity="high",
            )
        )

    if visitor.data_loader_calls == 0 and "DataLoader" in text:
        issues.append(
            AnalysisIssue(
                code="LOADER_WEAK",
                title="Inefficient DataLoader usage",
                detail="DataLoader appears in text but not as a clear call in the AST snapshot.",
                severity="low",
            )
        )
    elif visitor.data_loader_calls > 0 and not visitor.num_workers_values:
        issues.append(
            AnalysisIssue(
                code="LOADER_DEFAULT",
                title="Inefficient DataLoader usage",
                detail="DataLoader found without a literal num_workers= in this file.",
                severity="low",
            )
        )

    # Deduplicate by code while preserving order
    seen: set[str] = set()
    unique: list[AnalysisIssue] = []
    for issue in issues:
        if issue.code in seen:
            continue
        seen.add(issue.code)
        unique.append(issue)

    return AnalysisResult(file_path=path, issues=unique, raw_text=text)
