"""
analyzer.py — Analyze ML training scripts for inefficient patterns (AST + regex fallback).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from language_detector import detect_language


@dataclass
class AnalysisIssue:
    """One detected issue in the training script."""

    code: str
    title: str
    detail: str
    severity: str = "medium"


@dataclass
class AnalysisResult:
    """Full result of analyzing a single Python source."""

    file_path: Path
    issues: list[AnalysisIssue] = field(default_factory=list)
    raw_text: str = ""
    language: str = "Python"


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
        Parsed module tree, or None if syntax is invalid.
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
    if isinstance(node, ast.Num):  # pragma: no cover
        if isinstance(node.n, int):
            return int(node.n)
    return None


def _name_or_attr_str(node: ast.AST | None) -> str | None:
    """
    Best-effort string for a Name or Attribute chain (e.g. 'torch.optim.AdamW').
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
        """Initialize counters used during the AST walk."""
        self.batch_sizes: list[int] = []
        self.num_workers_values: list[int] = []
        self.data_loader_calls: int = 0
        self.mixed_precision_flags: list[bool] = []
        self.full_finetune_hints: list[str] = []
        self.optimizer_on_all_params: bool = False

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: ANN401
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
        """Pick up batch_size, num_workers, mixed_precision, and fine-tune flags."""
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
        Coarse signals for issue generation.
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


def _dedupe_issues(issues: list[AnalysisIssue]) -> list[AnalysisIssue]:
    """
    Deduplicate issues by issue.code while preserving first-seen order.
    """
    seen: set[str] = set()
    unique: list[AnalysisIssue] = []
    for issue in issues:
        if issue.code in seen:
            continue
        seen.add(issue.code)
        unique.append(issue)
    return unique


def _resource_heavy_detected(text: str) -> bool:
    """
    Heuristic detection of resource-heavy training patterns.

    Flags high epoch counts, oversized models, or repeated backward passes.
    """
    for pattern in (
        r"(?:num_epochs|max_epochs|epochs)\s*[=:]\s*(\d+)",
        r"for\s+epoch\s+in\s+range\s*\(\s*(\d+)",
    ):
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if int(m.group(1)) > 50:
                return True
    for pattern in (r"hidden_size\s*[=:]\s*(\d+)", r"n_embd\s*[=:]\s*(\d+)", r"d_model\s*[=:]\s*(\d+)"):
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if int(m.group(1)) >= 2048:
                return True
    if text.count(".backward()") >= 6:
        return True
    if len(re.findall(r"\.cuda\s*\(|\.to\s*\(\s*['\"]cuda", text, flags=re.IGNORECASE)) >= 8:
        return True
    return False


def _append_resource_heavy_issue(issues: list[AnalysisIssue], text: str, language: str) -> None:
    """Append RESOURCE_HEAVY issue when heuristics match."""
    if _resource_heavy_detected(text):
        issues.append(
            AnalysisIssue(
                code="RESOURCE_HEAVY",
                title="Resource heavy patterns",
                detail=f"High compute patterns detected in {language} source (epochs, model size, or GPU churn).",
                severity="high",
            )
        )


def _multilang_regex_signals(text: str, language: str) -> dict[str, Any]:
    """
    Regex heuristics for ML inefficiency patterns in non-Python languages.

    Args:
        text: Source code.
        language: Detected language label.

    Returns:
        Dict of coarse pattern flags.
    """
    patterns = {
        "batch_sizes": [],
        "workers_zero": False,
        "no_mixed_precision": False,
        "full_finetune": False,
        "loader_weak": False,
    }
    batch_re = r"(?:batch_size|batchSize|BATCH_SIZE)\s*[=:]\s*(\d+)"
    for m in re.finditer(batch_re, text, flags=re.IGNORECASE):
        patterns["batch_sizes"].append(int(m.group(1)))

    worker_re = r"(?:num_workers|numWorkers|NUM_WORKERS)\s*[=:]\s*(\d+)"
    for m in re.finditer(worker_re, text, flags=re.IGNORECASE):
        if int(m.group(1)) == 0:
            patterns["workers_zero"] = True

    if re.search(r"(?:mixed_precision|mixedPrecision|use_fp16|useFP16)\s*[=:]\s*(?:false|0)", text, flags=re.IGNORECASE):
        patterns["no_mixed_precision"] = True
    elif not re.search(r"(?:autocast|mixed.?precision|fp16|float16|half.?precision)", text, flags=re.IGNORECASE):
        patterns["no_mixed_precision"] = True

    if re.search(r"(?:train_full_model|fullFineTune|fineTuneAll|trainAll)\s*[=:]\s*(?:true|1)", text, flags=re.IGNORECASE):
        patterns["full_finetune"] = True
    if re.search(r"\.parameters\s*\(\s*\)|model\.train\s*\(\s*\)", text):
        patterns["full_finetune"] = True

    if "DataLoader" in text or "dataloader" in text.lower():
        if "num_workers" not in text and "numWorkers" not in text:
            patterns["loader_weak"] = True

    # language-specific hints
    if language == "C++" and re.search(r"std::thread::hardware_concurrency\s*\(\s*\)", text):
        patterns["loader_weak"] = False

    return patterns


def _analyze_multilang_text(text: str, logical_path: Path, language: str) -> AnalysisResult:
    """
    Analyze non-Python training-related source using regex heuristics.

    Args:
        text: Full source text.
        logical_path: Display path.
        language: Detected language name.

    Returns:
        AnalysisResult with issues list.
    """
    issues: list[AnalysisIssue] = []
    sig = _multilang_regex_signals(text, language)

    for bs in sig["batch_sizes"]:
        if bs > 128:
            issues.append(
                AnalysisIssue(
                    code="LARGE_BATCH",
                    title="Large batch size",
                    detail=f"Detected batch size {bs} (> 128) in {language} source.",
                    severity="medium",
                )
            )
    if sig["workers_zero"]:
        issues.append(
            AnalysisIssue(
                code="WORKERS_ZERO",
                title="DataLoader num_workers is 0",
                detail=f"Worker count is zero in {language} code — data loading may bottleneck training.",
                severity="medium",
            )
        )
    if sig["no_mixed_precision"]:
        issues.append(
            AnalysisIssue(
                code="NO_AMP",
                title="Mixed precision likely missing",
                detail=f"No clear mixed precision / FP16 patterns in {language} source.",
                severity="medium",
            )
        )
    if sig["full_finetune"]:
        issues.append(
            AnalysisIssue(
                code="FULL_FT",
                title="Full fine-tuning detected",
                detail=f"Full-parameter training patterns found in {language} source.",
                severity="high",
            )
        )
    if sig["loader_weak"]:
        issues.append(
            AnalysisIssue(
                code="LOADER_DEFAULT",
                title="Inefficient DataLoader usage",
                detail="Data loading without explicit worker tuning detected.",
                severity="low",
            )
        )

    _append_resource_heavy_issue(issues, text, language)

    return AnalysisResult(
        file_path=logical_path,
        issues=_dedupe_issues(issues),
        raw_text=text,
        language=language,
    )


def _analyze_text(text: str, logical_path: Path, language: str = "Python") -> AnalysisResult:
    """
    Core analysis: build an AnalysisResult from Python source text.

    Args:
        text: Full source code of a .py training script.
        logical_path: Display path (upload name or real file path).

    Returns:
        AnalysisResult with issues and raw text.
    """
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
                        detail=f"Detected batch_size = {bs} (> 128).",
                        severity="medium",
                    )
                )
        for nw in sig["num_workers"]:
            if nw == 0:
                issues.append(
                    AnalysisIssue(
                        code="WORKERS_ZERO",
                        title="DataLoader num_workers is 0",
                        detail="Try 2–8 workers to keep the GPU fed.",
                        severity="medium",
                    )
                )
        if sig["mixed_precision_false"]:
            issues.append(
                AnalysisIssue(
                    code="NO_AMP",
                    title="Mixed precision disabled",
                    detail="FP16/BF16 can reduce energy for many workloads.",
                    severity="medium",
                )
            )
        if sig["full_finetune"]:
            issues.append(
                AnalysisIssue(
                    code="FULL_FT",
                    title="Full fine-tuning detected",
                    detail="Updating all parameters increases compute.",
                    severity="high",
                )
            )
        if not sig["num_workers"] and "DataLoader" in text and "num_workers" not in text:
            issues.append(
                AnalysisIssue(
                    code="LOADER_DEFAULT",
                    title="Inefficient DataLoader usage",
                    detail="Consider explicit num_workers and pin_memory.",
                severity="low",
            )
        )
        _append_resource_heavy_issue(issues, text, language)
        return AnalysisResult(
            file_path=logical_path, issues=_dedupe_issues(issues), raw_text=text, language=language
        )

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
                detail="Use multiple workers so the GPU waits less on the CPU.",
                severity="medium",
            )
        )

    has_autocast_enabled_true = any(visitor.mixed_precision_flags)
    mixed_precision_literal_false = any(
        isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and "mixed" in t.id.lower() and "precision" in t.id.lower()
            for t in n.targets
        )
        and isinstance(n.value, ast.Constant)
        and n.value.value is False
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
    )

    if mixed_precision_literal_false or (not has_autocast_enabled_true and "autocast" not in text):
        if "torch.cuda.amp" in text or "autocast" in text:
            if mixed_precision_literal_false:
                issues.append(
                    AnalysisIssue(
                        code="NO_AMP",
                        title="Mixed precision disabled",
                        detail="AMP is referenced but mixed precision is set to False.",
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
                detail="Training all parameters is energy-heavy. Consider LoRA / adapters.",
                severity="high",
            )
        )

    if visitor.data_loader_calls == 0 and "DataLoader" in text:
        issues.append(
            AnalysisIssue(
                code="LOADER_WEAK",
                title="Inefficient DataLoader usage",
                detail="DataLoader appears in text but not as a clear call in the AST.",
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

    _append_resource_heavy_issue(issues, text, language)

    return AnalysisResult(
        file_path=logical_path, issues=_dedupe_issues(issues), raw_text=text, language=language
    )


def analyze_uploaded_source(source_code: str, filename: str = "uploaded.py") -> AnalysisResult:
    """
    Analyze in-memory source (e.g. from Streamlit file upload).

    Args:
        source_code: UTF-8 text of the script.
        filename: Original filename for display in reports.

    Returns:
        AnalysisResult for the given source.
    """
    language = detect_language(filename, source_code)
    path = Path(filename)
    if language == "Python":
        return _analyze_text(source_code, path, language=language)
    return _analyze_multilang_text(source_code, path, language)


def analyze_training_script(file_path: str | Path) -> AnalysisResult:
    """
    Analyze a Python training script on disk.

    Args:
        file_path: Path to the .py file.

    Returns:
        AnalysisResult containing issues and file text.

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
    language = detect_language(path.name, text)
    if language == "Python":
        return _analyze_text(text, path, language=language)
    return _analyze_multilang_text(text, path, language)
