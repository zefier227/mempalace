#!/usr/bin/env python3
"""
context_pack.py -- Context Pack: derive useful artifacts from a single text/dialogue.

MemPalace stays verbatim-first: the original text is ALWAYS the source of truth.
Derived artifacts are pointers and aids, never replacements.

Outputs:
  1. detailed_recap  -- BIG, detail-preserving recap (decisions, reasons,
     constraints, entities, chronology, open questions)
  2. wake_up         -- short starter context for a new session (reuses L1 style)
  3. aaak_text       -- AAAK-compressed text (reuses Dialect.compress)
  4. reusable_prompt -- ready-to-paste prompt for a new chat

LLM usage is OPTIONAL / BYO-LLM (same pattern as closet_llm.py).
Without an LLM, detailed_recap and reusable_prompt use template extraction.
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from .dialect import Dialect


LLM_HTTP_TIMEOUT = 120
LLM_MAX_RETRIES = 3

_DETAILED_RECAP_PROMPT = """You are producing a DETAILED RECAP of a text or dialogue.
The recap must be BIG and DENSE -- preserve nuances, not just the gist.

Source title: {title}
Source: {source}
Wing: {wing} | Room: {room}

CONTENT:
{content}

---

Produce a detailed recap in the SAME LANGUAGE as the content.
Structure it with these sections (omit any that have no content):

## Main Point
(1-2 sentences: the core idea or purpose)

## Decisions
- Decision: what was decided
  - Reason: why
  - Constraints: any constraints mentioned

## Key Entities
- People / projects / files / commands / errors mentioned

## Chronology
(ordered sequence of events, if present)

## Constraints & Limitations
(what can't be done, what must be avoided)

## Open Questions
(what is still unresolved)

## Important Details
(any nuances, edge cases, specific numbers, error messages, configs)

Rules:
- NEVER replace the original text -- this recap is a derivative aid.
- Preserve exact names, paths, error messages, and numbers.
- Keep it DENSE and LONG -- do not abbreviate for brevity.
- Write in the same language as the source content.
- Output plain text with markdown headings. No code fences. No commentary."""

_REUSABLE_PROMPT_TEMPLATE = """Based on the following context, I need you to help me continue working on this topic.
The context below is a detailed recap of a previous conversation/text.

{recap}

---

Key compressed reference (AAAK):
{aaak}

Please continue from where we left off. I may ask follow-up questions about any of the above."""


@dataclass
class ContextPackInput:
    raw_text: str
    title: str = ""
    source: str = ""
    wing: str = ""
    room: str = ""


@dataclass
class ContextPackMetadata:
    title: str = ""
    source: str = ""
    wing: str = ""
    room: str = ""
    original_chars: int = 0
    original_tokens_est: int = 0
    recap_tokens_est: int = 0
    wakeup_tokens_est: int = 0
    aaak_tokens_est: int = 0
    prompt_tokens_est: int = 0
    recap_method: str = "template"
    prompt_method: str = "template"
    generated_at: str = ""


@dataclass
class ContextPackResult:
    original_text: str
    detailed_recap: str
    wake_up: str
    aaak_text: str
    reusable_prompt: str
    metadata: ContextPackMetadata


class _LLMConfig:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.endpoint = (endpoint or os.environ.get("LLM_ENDPOINT", "")).rstrip("/")
        self.key = key or os.environ.get("LLM_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "")

    def missing(self) -> list:
        missing = []
        if not self.endpoint:
            missing.append("LLM_ENDPOINT")
        if not self.model:
            missing.append("LLM_MODEL")
        return missing

    def is_configured(self) -> bool:
        return not self.missing()


def _call_llm(cfg: _LLMConfig, prompt: str) -> Optional[str]:
    body = json.dumps(
        {
            "model": cfg.model,
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if cfg.key:
        headers["Authorization"] = f"Bearer {cfg.key}"

    url = f"{cfg.endpoint}/chat/completions"

    for attempt in range(LLM_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=LLM_HTTP_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw)
            text = payload["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return text
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < LLM_MAX_RETRIES - 1:
                time.sleep(2**attempt)
                continue
            return None
        except Exception:
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(2**attempt)
                continue
            return None
    return None


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def _extract_decision_lines(text: str) -> List[str]:
    decision_keywords = [
        "decided",
        "chose",
        "switched",
        "migrated",
        "replaced",
        "will use",
        "going with",
        "opted",
        "settled on",
        "решение",
        "решили",
        "выбрали",
        "перешли",
        "заменили",
    ]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 10:
            continue
        lower = stripped.lower()
        for kw in decision_keywords:
            if kw in lower:
                lines.append(stripped)
                break
    return lines[:15]


def _extract_entity_lines(text: str) -> List[str]:
    entities = set()
    cap_re = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
    for m in cap_re.finditer(text):
        name = m.group()
        stop = {"The This", "That These", "Those When", "In This", "If The"}
        if name not in stop and len(name) > 3:
            entities.add(name)
    code_re = re.compile(r"\b[A-Z_][A-Z0-9_]{2,}\b")
    for m in code_re.finditer(text):
        token = m.group()
        if len(token) > 2 and token not in {"THE", "AND", "FOR", "NOT", "BUT", "ALL"}:
            entities.add(token)
    path_re = re.compile(r"(?:[/\\][\w.-]+){2,}")
    for m in path_re.finditer(text):
        entities.add(m.group().strip("/\\"))
    return sorted(entities)[:20]


def _extract_chronology(text: str) -> List[str]:
    time_re = re.compile(
        r"(?:\d{1,2}[:.]\d{2}|\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{0,4}|\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{2,4})"
    )
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and time_re.search(stripped) and len(stripped) > 10:
            lines.append(stripped)
    return lines[:10]


def _extract_open_questions(text: str) -> List[str]:
    q_words = [
        "how do we",
        "how to",
        "what if",
        "should we",
        "can we",
        "is there",
        "неясно",
        "вопрос",
        "как",
        "нужно ли",
    ]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith("?") or stripped.endswith("?"):
            if len(stripped) > 8:
                lines.append(stripped)
                continue
        lower = stripped.lower()
        for kw in q_words:
            if kw in lower:
                lines.append(stripped)
                break
    return lines[:10]


def _extract_constraints(text: str) -> List[str]:
    constraint_kw = [
        "cannot",
        "can't",
        "must not",
        "limitation",
        "constraint",
        "restricted",
        "forbidden",
        "not allowed",
        "не можем",
        "нельзя",
        "ограничение",
        "запрещено",
        "недопустимо",
    ]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 10:
            continue
        lower = stripped.lower()
        for kw in constraint_kw:
            if kw in lower:
                lines.append(stripped)
                break
    return lines[:10]


def _template_detailed_recap(text: str, title: str, source: str, wing: str, room: str) -> str:
    decisions = _extract_decision_lines(text)
    entities = _extract_entity_lines(text)
    chronology = _extract_chronology(text)
    questions = _extract_open_questions(text)
    constraints = _extract_constraints(text)

    dialect = Dialect()
    key_sentence = dialect._extract_key_sentence(text)
    topics = dialect._extract_topics(text, max_topics=8)
    flags = dialect._detect_flags(text)
    emotions = dialect._detect_emotions(text)

    lines = []
    header_parts = []
    if title:
        header_parts.append(title)
    if source:
        header_parts.append(f"source: {source}")
    if wing:
        header_parts.append(f"wing: {wing}")
    if room:
        header_parts.append(f"room: {room}")
    if header_parts:
        lines.append("## Source")
        lines.append(" | ".join(header_parts))
        lines.append("")

    main_text = text.strip()
    if len(main_text) > 500:
        snippet = main_text[:500] + "..."
    else:
        snippet = main_text
    lines.append("## Main Point")
    lines.append(snippet)
    lines.append("")

    if topics:
        lines.append("## Topics")
        lines.append(", ".join(topics))
        lines.append("")

    if emotions:
        lines.append("## Emotional Tone")
        lines.append(", ".join(emotions))
        lines.append("")

    if flags:
        lines.append("## Flags")
        lines.append(", ".join(flags))
        lines.append("")

    if decisions:
        lines.append("## Decisions")
        for d in decisions:
            lines.append(f"- {d}")
        lines.append("")

    if entities:
        lines.append("## Key Entities")
        for e in entities:
            lines.append(f"- {e}")
        lines.append("")

    if chronology:
        lines.append("## Chronology")
        for c in chronology:
            lines.append(f"- {c}")
        lines.append("")

    if constraints:
        lines.append("## Constraints & Limitations")
        for c in constraints:
            lines.append(f"- {c}")
        lines.append("")

    if questions:
        lines.append("## Open Questions")
        for q in questions:
            lines.append(f"- {q}")
        lines.append("")

    if key_sentence:
        lines.append("## Key Quote")
        lines.append(f'"{key_sentence}"')
        lines.append("")

    return "\n".join(lines)


def _template_wake_up(text: str, title: str, wing: str) -> str:
    dialect = Dialect()
    topics = dialect._extract_topics(text, max_topics=5)
    entities = dialect._detect_entities_in_text(text)
    key_sentence = dialect._extract_key_sentence(text)
    emotions = dialect._detect_emotions(text)
    flags = dialect._detect_flags(text)

    lines = ["## WAKE-UP"]

    source_label = wing or title or "untitled"
    lines.append(f"Context: {source_label}")

    if entities:
        entity_str = "+".join(entities[:4])
        lines.append(f"Entities: {entity_str}")

    if topics:
        lines.append(f"Topics: {', '.join(topics)}")

    if emotions:
        lines.append(f"Tone: {', '.join(emotions)}")

    if flags:
        lines.append(f"Flags: {', '.join(flags)}")

    if key_sentence:
        lines.append(f'Anchor: "{key_sentence}"')

    snippet = text.strip().replace("\n", " ")
    if len(snippet) > 300:
        snippet = snippet[:297] + "..."
    lines.append(f"First 300 chars: {snippet}")

    return "\n".join(lines)


def _build_aaak(text: str, metadata: dict) -> str:
    dialect = Dialect()
    return dialect.compress(text, metadata=metadata)


def _build_reusable_prompt(recap: str, aaak: str, title: str) -> str:
    prompt = _REUSABLE_PROMPT_TEMPLATE.format(
        recap=recap,
        aaak=aaak,
    )
    if title:
        prompt = f"# Context: {title}\n\n{prompt}"
    return prompt


def build_context_pack(
    raw_text: str,
    title: str = "",
    source: str = "",
    wing: str = "",
    room: str = "",
    llm_config: Optional[_LLMConfig] = None,
    llm_callback: Optional[Callable[[str, str], Optional[str]]] = None,
) -> ContextPackResult:
    """Build a Context Pack from raw text.

    Args:
        raw_text: The source text/dialogue.
        title: Optional title.
        source: Optional source identifier.
        wing: Optional wing name for palace storage.
        room: Optional room name for palace storage.
        llm_config: Optional BYO-LLM config (same pattern as closet_llm.py).
        llm_callback: Optional callback(text, prompt) -> str for custom LLM calls.
                      Takes precedence over llm_config if both provided.

    Returns:
        ContextPackResult with all derived artifacts.
    """
    meta_for_aaak = {
        "source_file": source or title or "raw_text",
        "wing": wing,
        "room": room,
    }

    aaak_text = _build_aaak(raw_text, meta_for_aaak)

    wake_up = _template_wake_up(raw_text, title, wing)

    recap_method = "template"
    detailed_recap = _template_detailed_recap(raw_text, title, source, wing, room)

    if llm_callback is not None:
        prompt = _DETAILED_RECAP_PROMPT.format(
            title=title or "untitled",
            source=source or "raw text",
            wing=wing or "n/a",
            room=room or "n/a",
            content=raw_text[:30000],
        )
        llm_result = llm_callback(raw_text, prompt)
        if llm_result:
            detailed_recap = llm_result
            recap_method = "llm_callback"
    elif llm_config is not None and llm_config.is_configured():
        prompt = _DETAILED_RECAP_PROMPT.format(
            title=title or "untitled",
            source=source or "raw text",
            wing=wing or "n/a",
            room=room or "n/a",
            content=raw_text[:30000],
        )
        llm_result = _call_llm(llm_config, prompt)
        if llm_result:
            detailed_recap = llm_result
            recap_method = "llm_config"

    prompt_method = "template"
    reusable_prompt = _build_reusable_prompt(detailed_recap, aaak_text, title)

    if llm_callback is not None:
        rp_prompt = (
            "Generate a concise system prompt (under 200 words) that sets up a new AI session "
            "with the essential context from the following recap. "
            "The prompt should be ready to paste into a new chat. "
            "Write in the same language as the recap.\n\n"
            f"RECAP:\n{detailed_recap[:8000]}\n\n"
            f"AAAK REFERENCE:\n{aaak_text}"
        )
        llm_rp = llm_callback(raw_text, rp_prompt)
        if llm_rp:
            reusable_prompt = llm_rp
            prompt_method = "llm_callback"
    elif llm_config is not None and llm_config.is_configured():
        rp_prompt = (
            "Generate a concise system prompt (under 200 words) that sets up a new AI session "
            "with the essential context from the following recap. "
            "The prompt should be ready to paste into a new chat. "
            "Write in the same language as the recap.\n\n"
            f"RECAP:\n{detailed_recap[:8000]}\n\n"
            f"AAAK REFERENCE:\n{aaak_text}"
        )
        llm_rp = _call_llm(llm_config, rp_prompt)
        if llm_rp:
            reusable_prompt = llm_rp
            prompt_method = "llm_config"

    metadata = ContextPackMetadata(
        title=title,
        source=source,
        wing=wing,
        room=room,
        original_chars=len(raw_text),
        original_tokens_est=_estimate_tokens(raw_text),
        recap_tokens_est=_estimate_tokens(detailed_recap),
        wakeup_tokens_est=_estimate_tokens(wake_up),
        aaak_tokens_est=_estimate_tokens(aaak_text),
        prompt_tokens_est=_estimate_tokens(reusable_prompt),
        recap_method=recap_method,
        prompt_method=prompt_method,
        generated_at=datetime.now().isoformat(),
    )

    return ContextPackResult(
        original_text=raw_text,
        detailed_recap=detailed_recap,
        wake_up=wake_up,
        aaak_text=aaak_text,
        reusable_prompt=reusable_prompt,
        metadata=metadata,
    )


def save_context_pack_to_palace(
    result: ContextPackResult,
    palace_path: str,
) -> dict:
    """Save original text + derived artifacts to palace.

    Original text is stored as primary drawer.
    Derived artifacts are stored with metadata marking them as derived
    so they never replace the truth layer.

    Returns dict with counts.
    """
    from .palace import get_collection
    from .config import sanitize_name

    col = get_collection(palace_path, create=True)
    wing = result.metadata.wing or "context_pack"
    room = result.metadata.room or "recap"
    title = result.metadata.title or "untitled"
    source = result.metadata.source or title
    timestamp = datetime.now().isoformat()

    base_id = f"cp_{wing}_{room}_{sanitize_name(title)[:40]}"

    artifacts = {
        "original": (
            result.original_text,
            {
                "wing": wing,
                "room": room,
                "source_file": source,
                "chunk_index": 0,
                "filed_at": timestamp,
                "entities": "",
                "hall": "original",
                "normalize_version": 2,
                "context_pack": "true",
                "artifact_type": "original",
            },
        ),
        "detailed_recap": (
            result.detailed_recap,
            {
                "wing": wing,
                "room": f"{room}_recap",
                "source_file": source,
                "chunk_index": 0,
                "filed_at": timestamp,
                "entities": "",
                "hall": "recap",
                "normalize_version": 2,
                "context_pack": "true",
                "artifact_type": "detailed_recap",
            },
        ),
        "wake_up": (
            result.wake_up,
            {
                "wing": wing,
                "room": f"{room}_wakeup",
                "source_file": source,
                "chunk_index": 0,
                "filed_at": timestamp,
                "entities": "",
                "hall": "wake_up",
                "normalize_version": 2,
                "context_pack": "true",
                "artifact_type": "wake_up",
            },
        ),
        "aaak": (
            result.aaak_text,
            {
                "wing": wing,
                "room": f"{room}_aaak",
                "source_file": source,
                "chunk_index": 0,
                "filed_at": timestamp,
                "entities": "",
                "hall": "aaak",
                "normalize_version": 2,
                "context_pack": "true",
                "artifact_type": "aaak",
            },
        ),
        "reusable_prompt": (
            result.reusable_prompt,
            {
                "wing": wing,
                "room": f"{room}_prompt",
                "source_file": source,
                "chunk_index": 0,
                "filed_at": timestamp,
                "entities": "",
                "hall": "prompt",
                "normalize_version": 2,
                "context_pack": "true",
                "artifact_type": "reusable_prompt",
            },
        ),
    }

    filed = 0
    for name, (content, meta) in artifacts.items():
        doc_id = f"{base_id}_{name}"
        try:
            col.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[meta],
            )
            filed += 1
        except Exception:
            pass

    return {
        "filed": filed,
        "wing": wing,
        "room": room,
        "title": title,
    }


def context_pack_to_dict(result: ContextPackResult) -> dict:
    return {
        "original_text": result.original_text,
        "detailed_recap": result.detailed_recap,
        "wake_up": result.wake_up,
        "aaak_text": result.aaak_text,
        "reusable_prompt": result.reusable_prompt,
        "metadata": {
            "title": result.metadata.title,
            "source": result.metadata.source,
            "wing": result.metadata.wing,
            "room": result.metadata.room,
            "original_chars": result.metadata.original_chars,
            "original_tokens_est": result.metadata.original_tokens_est,
            "recap_tokens_est": result.metadata.recap_tokens_est,
            "wakeup_tokens_est": result.metadata.wakeup_tokens_est,
            "aaak_tokens_est": result.metadata.aaak_tokens_est,
            "prompt_tokens_est": result.metadata.prompt_tokens_est,
            "recap_method": result.metadata.recap_method,
            "prompt_method": result.metadata.prompt_method,
            "generated_at": result.metadata.generated_at,
        },
    }


def context_pack_to_json(result: ContextPackResult) -> str:
    return json.dumps(context_pack_to_dict(result), indent=2, ensure_ascii=False)
