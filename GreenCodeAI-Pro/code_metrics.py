"""
code_metrics.py — Line counts, structure metrics, and import counts for source files.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from language_detector import detect_language


@dataclass
class CodeMetrics:
    """Aggregated static metrics for one source file."""

    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    functions: int
    classes: int
    imports: int
    language: str


def _count_line_types(lines: list[str], language: str) -> tuple[int, int, int, int]:
    """
    Count total, code, comment, and blank lines using language-aware comment rules.

    Args:
        lines: Source split by newline.
        language: Detected language label.

    Returns:
        Tuple (total, code, comment, blank).
    """
    total = len(lines)
    blank = sum(1 for ln in lines if not ln.strip())
    comment = 0
    code = 0
    in_block = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if language == "Python":
            if line.startswith("#"):
                comment += 1
                continue
            code += 1
            continue
        if language in {"Java", "JavaScript", "TypeScript", "C++", "C", "C#", "Go", "PHP"}:
            if in_block:
                comment += 1
                if "*/" in line:
                    in_block = False
                continue
            if line.startswith("//"):
                comment += 1
                continue
            if line.startswith("/*"):
                comment += 1
                if "*/" not in line:
                    in_block = True
                continue
            code += 1
            continue
        if line.startswith("#") or line.startswith("//"):
            comment += 1
        else:
            code += 1

    return total, code, comment, blank


def _python_structure_counts(source: str) -> tuple[int, int, int]:
    """
    Count functions, classes, and imports in Python via AST (regex fallback).

    Returns:
        (functions, classes, imports)
    """
    try:
        tree = ast.parse(source)
        funcs = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        imports = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
        return funcs, classes, imports
    except SyntaxError:
        funcs = len(re.findall(r"^\s*def\s+\w+", source, flags=re.MULTILINE))
        classes = len(re.findall(r"^\s*class\s+\w+", source, flags=re.MULTILINE))
        imports = len(re.findall(r"^\s*(import |from \w+ import)", source, flags=re.MULTILINE))
        return funcs, classes, imports


def _generic_structure_counts(source: str, language: str) -> tuple[int, int, int]:
    """
    Regex-based structure counts for non-Python languages.

    Returns:
        (functions, classes, imports)
    """
    if language == "Java":
        funcs = len(re.findall(r"\b(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{", source))
        classes = len(re.findall(r"\bclass\s+\w+", source))
        imports = len(re.findall(r"^\s*import\s+", source, flags=re.MULTILINE))
    elif language in {"JavaScript", "TypeScript"}:
        funcs = len(re.findall(r"\bfunction\s+\w+|\b\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>", source))
        classes = len(re.findall(r"\bclass\s+\w+", source))
        imports = len(re.findall(r"^\s*import\s+", source, flags=re.MULTILINE))
    elif language == "C++":
        funcs = len(re.findall(r"\b[\w:<>\*&]+\s+\w+\s*\([^)]*\)\s*(?:const)?\s*\{", source))
        classes = len(re.findall(r"\b(?:class|struct)\s+\w+", source))
        imports = len(re.findall(r"^\s*#include\s+", source, flags=re.MULTILINE))
    elif language == "C":
        funcs = len(re.findall(r"\b[\w\s\*]+\s+\w+\s*\([^)]*\)\s*\{", source))
        classes = len(re.findall(r"\b(?:struct|enum)\s+\w+", source))
        imports = len(re.findall(r"^\s*#include\s+", source, flags=re.MULTILINE))
    elif language == "C#":
        funcs = len(re.findall(r"\b(?:public|private|protected|internal)?\s*(?:static\s+)?[\w<>\[\]]+\s+\w+\s*\([^)]*\)", source))
        classes = len(re.findall(r"\b(?:class|struct|interface)\s+\w+", source))
        imports = len(re.findall(r"^\s*using\s+", source, flags=re.MULTILINE))
    elif language == "Go":
        funcs = len(re.findall(r"\bfunc\s+(?:\([^)]+\)\s+)?\w+\s*\(", source))
        classes = len(re.findall(r"\btype\s+\w+\s+struct\b", source))
        imports = len(re.findall(r"^\s*import\s+", source, flags=re.MULTILINE))
    elif language == "PHP":
        funcs = len(re.findall(r"\bfunction\s+\w+\s*\(", source))
        classes = len(re.findall(r"\bclass\s+\w+", source))
        imports = len(re.findall(r"^\s*(?:use|require|include)\s+", source, flags=re.MULTILINE))
    else:
        funcs = classes = imports = 0
    return funcs, classes, imports


def compute_code_metrics(source: str, filename: str = "uploaded.py") -> CodeMetrics:
    """
    Compute line and structure metrics for a source file.

    Args:
        source: Full file text.
        filename: Original filename (used for language detection).

    Returns:
        CodeMetrics dataclass.
    """
    language = detect_language(filename, source)
    lines = source.splitlines()
    total, code, comment, blank = _count_line_types(lines, language)

    if language == "Python":
        funcs, classes, imports = _python_structure_counts(source)
    else:
        funcs, classes, imports = _generic_structure_counts(source, language)

    return CodeMetrics(
        total_lines=total,
        code_lines=code,
        comment_lines=comment,
        blank_lines=blank,
        functions=funcs,
        classes=classes,
        imports=imports,
        language=language,
    )
