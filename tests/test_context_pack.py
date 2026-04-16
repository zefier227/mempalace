"""
test_context_pack.py -- Tests for the Context Pack feature.

Covers:
  - Core flow: build_context_pack with template extraction
  - AAAK output (reuses Dialect.compress)
  - Wake-up output
  - Detailed recap structure and content preservation
  - Reusable prompt generation
  - Save to palace behavior
  - CLI smoke test
  - Long Russian text / dialogue case
  - JSON/structured output
"""

import json
import os
import argparse
from unittest.mock import patch

import pytest

from mempalace.context_pack import (
    build_context_pack,
    save_context_pack_to_palace,
    context_pack_to_dict,
    context_pack_to_json,
    _LLMConfig,
    _estimate_tokens,
    _extract_decision_lines,
    _extract_entity_lines,
    _extract_chronology,
    _extract_open_questions,
    _extract_constraints,
)


SAMPLE_TEXT = """We decided to use GraphQL instead of REST for the API layer.
The reason is that our frontend team needs flexible querying capabilities
and REST endpoints were becoming too chatty.

Alice suggested we use Apollo Server, but Bob preferred Relay.
We went with Apollo because it has better TypeScript support.

Key constraints:
- Cannot use WebSocket subscriptions (infrastructure limitation)
- Must support backward compatibility with REST clients for 6 months

Open question: should we migrate the auth module to passkeys by Q3?

Error: AUTH_TIMEOUT when refreshing JWT tokens under load.
File: src/auth/token_refresh.py
Command: npm run migrate:auth
"""

RUSSIAN_DIALOGUE = """В: Давайте обсудим архитектуру нового модуля памяти.
О: Предлагаю использовать метод Zettelkasten — маленькие перекрёстные карточки.
В: Почему не просто векторная база данных?
О: Потому что векторный поиск теряет контекст. Zettel сохраняет связи между идеями.

Решение: реализуем 4-слойную систему памяти (L0-L3).
Причина: каждый слой загружает только нужный объём контекста.
Ограничение: L1 не должен превышать 800 токенов.

В: Как быть с русскоязычным контентом?
О: AAAK работает с любым языком, но нужно добавить i18n-паттерны.

Ошибка: CYPRESS_TIMEOUT при тестировании миграции.
Файл: tests/migration.spec.ts
Команда: pytest tests/ -k migration

Остаются вопросы:
- Нужно ли добавлять поддержку китайского языка?
- Какую модель эмбеддингов использовать для кириллицы?
"""


class TestBuildContextPackCore:
    def test_basic_build(self):
        result = build_context_pack(SAMPLE_TEXT)
        assert result.original_text == SAMPLE_TEXT
        assert isinstance(result.detailed_recap, str)
        assert len(result.detailed_recap) > 0
        assert isinstance(result.wake_up, str)
        assert len(result.wake_up) > 0
        assert isinstance(result.aaak_text, str)
        assert len(result.aaak_text) > 0
        assert isinstance(result.reusable_prompt, str)
        assert len(result.reusable_prompt) > 0

    def test_original_text_preserved_verbatim(self):
        result = build_context_pack(SAMPLE_TEXT)
        assert result.original_text == SAMPLE_TEXT

    def test_metadata_populated(self):
        result = build_context_pack(
            SAMPLE_TEXT, title="Test", source="test.txt", wing="proj", room="arch"
        )
        m = result.metadata
        assert m.title == "Test"
        assert m.source == "test.txt"
        assert m.wing == "proj"
        assert m.room == "arch"
        assert m.original_chars == len(SAMPLE_TEXT)
        assert m.original_tokens_est > 0
        assert m.recap_tokens_est > 0
        assert m.generated_at != ""

    def test_aaak_reuses_dialect_compress(self):
        result = build_context_pack("We decided to use GraphQL.", wing="project")
        assert "|" in result.aaak_text

    def test_wakeup_has_structure(self):
        result = build_context_pack(SAMPLE_TEXT, title="Architecture Decision")
        assert "WAKE-UP" in result.wake_up
        assert "Architecture Decision" in result.wake_up

    def test_detailed_recap_preserves_decisions(self):
        result = build_context_pack(SAMPLE_TEXT)
        lower = result.detailed_recap.lower()
        assert "decided" in lower or "graphql" in lower

    def test_detailed_recap_is_big(self):
        result = build_context_pack(SAMPLE_TEXT)
        assert len(result.detailed_recap) > len(SAMPLE_TEXT) * 0.3

    def test_reusable_prompt_contains_recap(self):
        result = build_context_pack(SAMPLE_TEXT)
        assert len(result.reusable_prompt) > 50
        assert (
            "context" in result.reusable_prompt.lower() or "recap" in result.reusable_prompt.lower()
        )

    def test_empty_text(self):
        result = build_context_pack("")
        assert result.original_text == ""
        assert result.metadata.original_chars == 0


class TestTemplateExtractors:
    def test_extract_decisions_english(self):
        lines = _extract_decision_lines(SAMPLE_TEXT)
        assert any("decided" in line.lower() for line in lines)
        assert any("graphql" in line.lower() for line in lines)

    def test_extract_decisions_russian(self):
        lines = _extract_decision_lines(RUSSIAN_DIALOGUE)
        assert len(lines) > 0

    def test_extract_entities(self):
        lines = _extract_entity_lines("Changed Apollo Server config and relay_settings.json")
        assert len(lines) > 0

    def test_extract_entities_paths(self):
        lines = _extract_entity_lines("Changed src/auth/token_refresh.py and config/settings.json")
        assert any("token" in line or "auth" in line for line in lines)

    def test_extract_chronology(self):
        text = "On 2024-03-15 we launched v2. Then on 2024-04-01 we migrated."
        lines = _extract_chronology(text)
        assert len(lines) > 0

    def test_extract_open_questions(self):
        lines = _extract_open_questions(SAMPLE_TEXT)
        assert any("passkeys" in line.lower() or "migrate" in line.lower() for line in lines)

    def test_extract_constraints(self):
        lines = _extract_constraints(SAMPLE_TEXT)
        assert any("cannot" in line.lower() or "limitation" in line.lower() for line in lines)


class TestRussianDialogue:
    def test_build_russian(self):
        result = build_context_pack(RUSSIAN_DIALOGUE, title="Архитектура памяти", wing="mempalace")
        assert result.original_text == RUSSIAN_DIALOGUE
        assert len(result.detailed_recap) > 100
        assert "WAKE-UP" in result.wake_up
        assert "|" in result.aaak_text

    def test_russian_decisions_detected(self):
        lines = _extract_decision_lines(RUSSIAN_DIALOGUE)
        assert len(lines) > 0

    def test_russian_questions_detected(self):
        lines = _extract_open_questions(RUSSIAN_DIALOGUE)
        assert len(lines) > 0


class TestLLMConfig:
    def test_missing_when_empty(self):
        cfg = _LLMConfig()
        assert len(cfg.missing()) >= 2

    def test_configured_when_set(self):
        cfg = _LLMConfig(endpoint="http://localhost:11434/v1", model="llama3")
        assert cfg.is_configured()

    def test_key_optional(self):
        cfg = _LLMConfig(endpoint="http://localhost:11434/v1", model="llama3")
        assert "LLM_KEY" not in cfg.missing()

    def test_env_override(self):
        with patch.dict(os.environ, {"LLM_ENDPOINT": "http://test/v1", "LLM_MODEL": "test-model"}):
            cfg = _LLMConfig()
            assert cfg.endpoint == "http://test/v1"
            assert cfg.model == "test-model"


class TestLLMCallback:
    def test_llm_callback_overrides_recap(self):
        def my_callback(text, prompt):
            return "CUSTOM RECAP FROM LLM"

        result = build_context_pack(SAMPLE_TEXT, llm_callback=my_callback)
        assert result.detailed_recap == "CUSTOM RECAP FROM LLM"
        assert result.metadata.recap_method == "llm_callback"

    def test_llm_callback_returns_none_falls_back(self):
        def my_callback(text, prompt):
            return None

        result = build_context_pack(SAMPLE_TEXT, llm_callback=my_callback)
        assert result.metadata.recap_method == "template"
        assert len(result.detailed_recap) > 0

    def test_llm_callback_for_prompt(self):
        def my_callback(text, prompt):
            if "system prompt" in prompt.lower():
                return "CUSTOM SYSTEM PROMPT"
            return "CUSTOM RECAP"

        result = build_context_pack(SAMPLE_TEXT, llm_callback=my_callback)
        assert result.reusable_prompt == "CUSTOM SYSTEM PROMPT"
        assert result.metadata.prompt_method == "llm_callback"


class TestSaveToPalace:
    def test_save_creates_drawers(self, palace_path):
        result = build_context_pack(SAMPLE_TEXT, title="Test", wing="proj", room="arch")
        save_result = save_context_pack_to_palace(result, palace_path)
        assert save_result["filed"] == 5
        assert save_result["wing"] == "proj"

    def test_save_original_is_primary(self, palace_path):
        result = build_context_pack(SAMPLE_TEXT, title="Test", wing="proj")
        save_context_pack_to_palace(result, palace_path)
        from mempalace.palace import get_collection

        col = get_collection(palace_path, create=False)
        all_data = col.get(include=["metadatas", "documents"])
        originals = [
            (m, d)
            for m, d in zip(all_data["metadatas"], all_data["documents"])
            if m.get("artifact_type") == "original"
        ]
        assert len(originals) == 1
        assert originals[0][1] == SAMPLE_TEXT

    def test_derived_artifacts_labeled(self, palace_path):
        result = build_context_pack(SAMPLE_TEXT, title="Test", wing="proj")
        save_context_pack_to_palace(result, palace_path)
        from mempalace.palace import get_collection

        col = get_collection(palace_path, create=False)
        all_data = col.get(include=["metadatas"])
        artifact_types = {m.get("artifact_type") for m in all_data["metadatas"]}
        assert "original" in artifact_types
        assert "detailed_recap" in artifact_types
        assert "wake_up" in artifact_types
        assert "aaak" in artifact_types
        assert "reusable_prompt" in artifact_types


class TestStructuredOutput:
    def test_to_dict(self):
        result = build_context_pack(SAMPLE_TEXT, title="Test")
        d = context_pack_to_dict(result)
        assert "original_text" in d
        assert "detailed_recap" in d
        assert "wake_up" in d
        assert "aaak_text" in d
        assert "reusable_prompt" in d
        assert "metadata" in d

    def test_to_json(self):
        result = build_context_pack(SAMPLE_TEXT, title="Test")
        j = context_pack_to_json(result)
        parsed = json.loads(j)
        assert parsed["metadata"]["title"] == "Test"

    def test_json_russian_preserved(self):
        result = build_context_pack(RUSSIAN_DIALOGUE, title="Русский тест")
        j = context_pack_to_json(result)
        assert "Русский тест" in j


class TestCLI:
    def test_cmd_context_pack_text(self):
        from mempalace.cli import cmd_context_pack

        args = argparse.Namespace(
            file=None,
            text=SAMPLE_TEXT,
            title="Test",
            source=None,
            wing=None,
            room=None,
            section="all",
            json=False,
            save=False,
            no_save_hint=True,
            llm=False,
            llm_endpoint=None,
            llm_model=None,
            llm_key=None,
            palace=None,
        )
        with patch("mempalace.cli.MempalaceConfig") as mock_cfg:
            mock_cfg.return_value.palace_path = "/tmp/test_palace"
            cmd_context_pack(args)

    def test_cmd_context_pack_json_output(self, capsys):
        from mempalace.cli import cmd_context_pack

        args = argparse.Namespace(
            file=None,
            text="Simple test text.",
            title="JSON Test",
            source=None,
            wing=None,
            room=None,
            section="all",
            json=True,
            save=False,
            no_save_hint=True,
            llm=False,
            llm_endpoint=None,
            llm_model=None,
            llm_key=None,
            palace=None,
        )
        with patch("mempalace.cli.MempalaceConfig") as mock_cfg:
            mock_cfg.return_value.palace_path = "/tmp/test_palace"
            cmd_context_pack(args)
            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            assert parsed["metadata"]["title"] == "JSON Test"

    def test_cmd_context_pack_single_section(self, capsys):
        from mempalace.cli import cmd_context_pack

        args = argparse.Namespace(
            file=None,
            text=SAMPLE_TEXT,
            title=None,
            source=None,
            wing=None,
            room=None,
            section="aaak",
            json=False,
            save=False,
            no_save_hint=True,
            llm=False,
            llm_endpoint=None,
            llm_model=None,
            llm_key=None,
            palace=None,
        )
        with patch("mempalace.cli.MempalaceConfig") as mock_cfg:
            mock_cfg.return_value.palace_path = "/tmp/test_palace"
            cmd_context_pack(args)
            captured = capsys.readouterr()
            assert "AAAK COMPRESSED" in captured.out

    def test_cmd_context_pack_file_not_found(self):
        from mempalace.cli import cmd_context_pack

        args = argparse.Namespace(
            file="/nonexistent/file.txt",
            text=None,
            title=None,
            source=None,
            wing=None,
            room=None,
            section="all",
            json=False,
            save=False,
            no_save_hint=True,
            llm=False,
            llm_endpoint=None,
            llm_model=None,
            llm_key=None,
            palace=None,
        )
        with pytest.raises(SystemExit):
            cmd_context_pack(args)


class TestEstimateTokens:
    def test_basic(self):
        assert _estimate_tokens("hello world") > 0

    def test_empty(self):
        assert _estimate_tokens("") >= 1

    def test_russian(self):
        tokens = _estimate_tokens(RUSSIAN_DIALOGUE)
        assert tokens > 10
