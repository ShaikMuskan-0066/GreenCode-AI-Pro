"""
ai_assistant.py — Sustainability-focused AI assistant response generation and context handling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssistantContext:
    """Session-derived context for personalized sustainability guidance."""

    has_analysis: bool = False
    file_name: str = ""
    language: str = ""
    green_score: int | None = None
    green_status: str = ""
    energy_kwh: float | None = None
    co2_kg: float | None = None
    cost_inr: float | None = None
    issue_titles: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metrics_summary: str = ""
    quality_summary: str = ""
    comparison_note: str = ""
    repo_note: str = ""


def _session_get(session: Any, key: str, default: Any = None) -> Any:
    """Read a key from Streamlit session_state or a plain dict."""
    if hasattr(session, "get"):
        return session.get(key, default)
    return getattr(session, key, default)


def build_assistant_context(session: Any) -> AssistantContext:
    """
    Build assistant context from Streamlit session state (read-only).

    Args:
        session: ``st.session_state`` or dict-like session object.

    Returns:
        AssistantContext populated from the latest dashboard analysis when available.
    """
    ctx = AssistantContext()

    analysis = _session_get(session, "analysis")
    carbon = _session_get(session, "carbon_base")
    if analysis is not None and carbon is not None:
        ctx.has_analysis = True
        ctx.file_name = str(getattr(analysis, "file_path", "") or "")
        ctx.language = str(_session_get(session, "language") or getattr(analysis, "language", "") or "")
        ctx.green_score = _session_get(session, "green_score")
        if ctx.green_score is not None:
            try:
                from sustainability_score import score_status_label

                ctx.green_status = score_status_label(int(ctx.green_score))
            except Exception:  # noqa: BLE001
                ctx.green_status = ""
        ctx.energy_kwh = float(getattr(carbon, "energy_kwh", 0) or 0)
        ctx.co2_kg = float(getattr(carbon, "co2_kg", 0) or 0)
        ctx.cost_inr = float(getattr(carbon, "cost_inr", 0) or 0)
        issues = getattr(analysis, "issues", []) or []
        ctx.issue_titles = [getattr(i, "title", str(i)) for i in issues]
        ctx.suggestions = list(_session_get(session, "suggestions") or [])

        metrics = _session_get(session, "code_metrics")
        if metrics is not None:
            ctx.metrics_summary = (
                f"{getattr(metrics, 'total_lines', 0)} lines, "
                f"{getattr(metrics, 'functions', 0)} functions, "
                f"{getattr(metrics, 'imports', 0)} imports"
            )

        quality = _session_get(session, "quality_insights")
        if quality is not None:
            ctx.quality_summary = (
                f"maintainability {getattr(quality, 'maintainability_score', '—')}/100, "
                f"readability {getattr(quality, 'readability_score', '—')}/100"
            )

    compare_a = _session_get(session, "compare_a")
    compare_b = _session_get(session, "compare_b")
    if compare_a and compare_b:
        ctx.comparison_note = (
            f"Recent comparison: {compare_a.get('name', 'File A')} (score {compare_a.get('score', '—')}) "
            f"vs {compare_b.get('name', 'File B')} (score {compare_b.get('score', '—')})."
        )

    repo = _session_get(session, "repo_result")
    if repo is not None and not getattr(repo, "error", None):
        meta = getattr(repo, "metadata", None)
        repo_name = getattr(meta, "name", "") if meta else ""
        ctx.repo_note = (
            f"Recent repo scan: {repo_name or 'repository'} — "
            f"sustainability {getattr(repo, 'sustainability_score', '—')}/100, "
            f"grade {getattr(repo, 'sustainability_grade', '—')}, "
            f"{getattr(repo, 'aggregate_issues', 0)} issues."
        )

    return ctx


def _context_block(ctx: AssistantContext) -> str:
    """Format session context for LLM system prompts."""
    if not ctx.has_analysis and not ctx.comparison_note and not ctx.repo_note:
        return "No active file analysis in session. Give general green-AI guidance."

    lines = ["## User session context"]
    if ctx.has_analysis:
        lines.append(f"- Analyzed file: `{ctx.file_name}` ({ctx.language})")
        if ctx.green_score is not None:
            lines.append(f"- Sustainability score: **{ctx.green_score}/100** ({ctx.green_status})")
        if ctx.co2_kg is not None:
            lines.append(
                f"- Estimated impact: **{ctx.co2_kg:.4f} kg CO₂**, "
                f"**{ctx.energy_kwh:.4f} kWh**, **₹{ctx.cost_inr:.2f}**"
            )
        if ctx.issue_titles:
            lines.append(f"- Detected issues: {', '.join(ctx.issue_titles)}")
        if ctx.suggestions:
            lines.append(f"- Platform suggestions: {'; '.join(ctx.suggestions[:6])}")
        if ctx.metrics_summary:
            lines.append(f"- Code metrics: {ctx.metrics_summary}")
        if ctx.quality_summary:
            lines.append(f"- Quality: {ctx.quality_summary}")
    if ctx.comparison_note:
        lines.append(f"- {ctx.comparison_note}")
    if ctx.repo_note:
        lines.append(f"- {ctx.repo_note}")
    lines.append(
        "\nUse this context when relevant. Prioritize carbon reduction, energy efficiency, "
        "and green software practices."
    )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are **GreenCode AI Pro Assistant** — a sustainability engineering tutor for ML training scripts.

Your expertise:
- Carbon impact and energy usage of AI training
- Green software practices for PyTorch/TensorFlow workflows
- LoRA / PEFT, quantization, mixed precision (AMP), DataLoader tuning, batch size optimization
- Translating detected inefficiencies into actionable fixes

Rules:
- Be concise, professional, and actionable (3–6 short paragraphs or bullets max unless asked for depth).
- When session context is provided, reference the user's actual scores, issues, and suggestions.
- Never invent metrics not in context; if unknown, say what to analyze on the Dashboard.
- Focus on sustainability outcomes: lower CO₂, lower kWh, better resource efficiency.
- Use markdown for readability."""


def _normalize_history(history: list[dict] | None) -> list[dict]:
    """Keep last N turns for API context window."""
    if not history:
        return []
    cleaned: list[dict] = []
    for msg in history:
        role = msg.get("role", "")
        content = str(msg.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            cleaned.append({"role": role, "content": content})
    return cleaned[-10:]


def _try_openai(question: str, history: list[dict], ctx: AssistantContext) -> str | None:
    """Call OpenAI if API key is configured."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + _context_block(ctx)},
            *_normalize_history(history),
            {"role": "user", "content": question},
        ]
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=600,
            temperature=0.4,
        )
        text = resp.choices[0].message.content
        return text.strip() if text else None
    except Exception:  # noqa: BLE001
        return None


def _try_gemini(question: str, history: list[dict], ctx: AssistantContext) -> str | None:
    """Call Gemini if API key is configured."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)

        hist_text = ""
        for msg in _normalize_history(history):
            prefix = "User" if msg["role"] == "user" else "Assistant"
            hist_text += f"{prefix}: {msg['content']}\n"

        prompt = (
            f"{SYSTEM_PROMPT}\n\n{_context_block(ctx)}\n\n"
            f"Conversation so far:\n{hist_text}\nUser: {question}\n\nAssistant:"
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        return text.strip() if text else None
    except Exception:  # noqa: BLE001
        return None


def _match_topic(question: str) -> str | None:
    """Return topic key from question keywords."""
    q = question.lower()
    topics: list[tuple[str, list[str]]] = [
        ("carbon_high", ["carbon", "co2", "emission", "footprint"]),
        ("energy", ["energy", "kwh", "electricity", "power"]),
        ("lora", ["lora", "peft", "adapter", "adapters"]),
        ("quantization", ["quantiz", "int8", "4-bit", "4bit", "8-bit"]),
        ("mixed_precision", ["mixed precision", "amp", "fp16", "bf16", "autocast"]),
        ("batch_size", ["batch size", "batch_size", "large batch"]),
        ("dataloader", ["dataloader", "num_workers", "workers", "pin_memory"]),
        ("full_finetune", ["full fine", "full fine-tun", "finetune all", "full parameter"]),
        ("resource", ["resource", "gpu", "memory", "utilization"]),
        ("score", ["green score", "sustainability score", "grade"]),
        ("suggestions", ["suggest", "optimize", "improve", "fix", "recommend"]),
        ("compare", ["compare", "baseline", "candidate", "which file"]),
        ("github", ["github", "repository", "repo"]),
    ]
    for key, keywords in topics:
        if any(kw in q for kw in keywords):
            return key
    return None


def _personalized_opening(ctx: AssistantContext) -> str:
    """One-line personalization when analysis exists."""
    if not ctx.has_analysis:
        return ""
    score_part = f" (score **{ctx.green_score}/100**)" if ctx.green_score is not None else ""
    return f"For your analyzed script `{ctx.file_name}`{score_part}:\n\n"


def _local_reply(question: str, ctx: AssistantContext, history: list[dict]) -> str:
    """Rule-based sustainability tutor with session-aware responses."""
    q = question.strip().lower()
    if not q:
        return (
            "Ask me about **carbon impact**, **energy usage**, **LoRA**, **mixed precision**, "
            "**quantization**, **DataLoader workers**, or how to improve your **sustainability score**.\n\n"
            "Tip: run an analysis on the Dashboard first — I'll use your real scores and detected issues."
        )

    topic = _match_topic(question)
    opening = _personalized_opening(ctx)

    if topic == "carbon_high" or ("carbon" in q and ("high" in q or "why" in q)):
        body = (
            "**Why carbon usage may be high:**\n"
            "- **Large batch sizes** increase peak GPU power draw\n"
            "- **Full fine-tuning** updates all weights — far more compute than LoRA/PEFT\n"
            "- **Missing mixed precision** (FP16/BF16) wastes energy per training step\n"
            "- **`num_workers = 0`** leaves the GPU idle waiting on CPU data loading\n"
            "- **Long training duration** multiplies total kWh and CO₂\n"
        )
        if ctx.issue_titles:
            body += f"\n**Your detected issues:** {', '.join(ctx.issue_titles)}\n"
        if ctx.co2_kg is not None and ctx.has_analysis:
            body += (
                f"\n**Your current estimate:** {ctx.co2_kg:.4f} kg CO₂, "
                f"{ctx.energy_kwh:.4f} kWh (₹{ctx.cost_inr:.2f})."
            )
        return opening + body

    if topic == "lora":
        extra = ""
        if ctx.has_analysis and any("fine" in t.lower() for t in ctx.issue_titles):
            extra = "\n\nYour analysis flagged **full fine-tuning** — LoRA is the highest-impact fix to try first."
        return opening + (
            "**LoRA (Low-Rank Adaptation)** trains small adapter matrices instead of all model weights.\n\n"
            "- Cuts **memory** and **compute** dramatically vs full fine-tuning\n"
            "- Often reaches similar task quality with a fraction of energy\n"
            "- Pair with **mixed precision** for additional savings"
            + extra
        )

    if topic == "quantization":
        return opening + (
            "**Quantization** stores weights in lower precision (INT8, 4-bit) to reduce memory and speed up inference/training.\n\n"
            "- Lower memory → smaller GPUs or larger effective batch sizes\n"
            "- Less data movement → lower energy per step\n"
            "- Use when quality tolerance allows (many deployment scenarios)"
        )

    if topic == "mixed_precision":
        return opening + (
            "**Mixed precision (AMP)** runs many ops in FP16/BF16 while keeping sensitive ops in FP32.\n\n"
            "- Higher throughput on modern GPUs\n"
            "- **Lower energy per training step** in most PyTorch workflows\n"
            "- Enable via `torch.cuda.amp.autocast` + `GradScaler`"
            + (
                "\n\nYour script shows **mixed precision issues** — enabling AMP is a quick win."
                if ctx.has_analysis and any("mixed" in t.lower() for t in ctx.issue_titles)
                else ""
            )
        )

    if topic == "batch_size":
        return opening + (
            "**Batch size** trades memory, convergence speed, and power draw.\n\n"
            "- Oversized batches spike **peak power** and can hurt generalization\n"
            "- **Reduce batch size** or use **gradient accumulation** for similar effective batch with smoother energy\n"
            "- Target the smallest batch that maintains stable training"
            + (
                "\n\nYour analysis detected a **large batch size** — try halving it and measuring CO₂ on the Dashboard."
                if ctx.has_analysis and any("batch" in t.lower() for t in ctx.issue_titles)
                else ""
            )
        )

    if topic == "dataloader":
        return opening + (
            "**DataLoader tuning** keeps the GPU fed and reduces idle energy:\n\n"
            "- Set `num_workers` to **2–8** (scale with CPU cores)\n"
            "- Use `pin_memory=True` on CUDA\n"
            "- Prefetch batches so the GPU isn't waiting on disk/CPU"
            + (
                "\n\nYour script has **`num_workers = 0`** or weak DataLoader config — fix this before optimizing the model."
                if ctx.has_analysis and any("worker" in t.lower() or "dataloader" in t.lower() for t in ctx.issue_titles)
                else ""
            )
        )

    if topic == "full_finetune":
        return opening + (
            "**Full fine-tuning** updates every parameter — the most energy-intensive training mode.\n\n"
            "Prefer **LoRA / PEFT adapters** when task quality allows. Combine with **mixed precision** and **smaller batches**."
        )

    if topic == "energy" or ("reduce" in q and "energy" in q):
        steps = [
            "Enable **mixed precision** (AMP)",
            "Use **LoRA** instead of full fine-tuning",
            "Tune **DataLoader workers** (`num_workers > 0`, `pin_memory=True`)",
            "Reduce **oversized batch sizes**",
            "Apply **quantization** where quality allows",
            "Profile GPU utilization — eliminate idle/wait time",
        ]
        body = "**Reduce training energy:**\n" + "\n".join(f"- {s}" for s in steps)
        if ctx.suggestions:
            body += "\n\n**For your file, start with:**\n" + "\n".join(f"- {s}" for s in ctx.suggestions[:5])
        return opening + body

    if topic == "score":
        if ctx.has_analysis and ctx.green_score is not None:
            return opening + (
                f"Your **sustainability score** is **{ctx.green_score}/100** ({ctx.green_status}).\n\n"
                "Scores drop with higher CO₂/kWh estimates and more detected green-practice violations. "
                "Address issues in order: mixed precision → DataLoader → batch size → LoRA vs full FT."
            )
        return (
            "Run a script analysis on the **Dashboard** to get your sustainability score. "
            "I'll then explain what's pulling it down and how to improve it."
        )

    if topic == "suggestions":
        if ctx.suggestions:
            items = "\n".join(f"- **{s}**" for s in ctx.suggestions)
            return opening + f"**Optimization path for your script:**\n{items}"
        return opening + (
            "**General priority order:**\n"
            "- Use LoRA\n"
            "- Enable Mixed Precision\n"
            "- Improve DataLoader Workers\n"
            "- Reduce Batch Size\n"
            "- Apply Quantization\n"
            "- Optimize Resource Usage"
        )

    if topic == "compare" and ctx.comparison_note:
        return f"**From your recent comparison:**\n{ctx.comparison_note}\n\nChoose the higher sustainability score and lower CO₂ candidate, then apply its patterns to production training."

    if topic == "github" and ctx.repo_note:
        return f"**From your repository scan:**\n{ctx.repo_note}\n\nFocus on training scripts with the most issues — apply LoRA, AMP, and DataLoader fixes file by file."

    if topic == "resource":
        return opening + (
            "**Resource efficiency** means less wasted GPU/CPU cycles per useful training step.\n\n"
            "- Avoid redundant forward/backward passes\n"
            "- Don't load full models when adapters suffice\n"
            "- Monitor utilization — sub-50% GPU usage often means data loading or batching bottlenecks"
        )

    # Follow-up awareness from recent history
    if history:
        last_user = next((m["content"].lower() for m in reversed(history) if m.get("role") == "user"), "")
        if "carbon" in last_user and ("how" in q or "fix" in q):
            return _local_reply("how to reduce energy", ctx, [])

    return (
        opening
        + "I'm your **GreenCode AI sustainability assistant**. I can help with:\n"
        "- Why **carbon / energy** usage is high\n"
        "- **LoRA**, **quantization**, **mixed precision**, **DataLoader** tuning\n"
        "- Interpreting your **sustainability score** and detected issues\n\n"
        + (
            f"Your session has analysis for `{ctx.file_name}` — ask how to fix specific detected issues."
            if ctx.has_analysis
            else "Analyze a script on the Dashboard first for personalized guidance."
        )
    )


def generate_assistant_reply(
    question: str,
    history: list[dict] | None = None,
    context: AssistantContext | None = None,
) -> str:
    """
    Generate an assistant reply using API backends or local sustainability knowledge.

    Args:
        question: User message.
        history: Prior chat turns ``[{"role": "user"|"assistant", "content": "..."}]``.
        context: Optional session context from ``build_assistant_context``.

    Returns:
        Markdown-formatted assistant response.
    """
    ctx = context or AssistantContext()
    hist = _normalize_history(history)

    for backend in (_try_openai, _try_gemini):
        reply = backend(question, hist, ctx)
        if reply:
            return reply

    return _local_reply(question, ctx, hist)
