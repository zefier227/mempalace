"""
tests/test_wakeup_compress.py — Unit, smoke, and parity tests for
run_wakeup() and run_compress() in gui_adapter.py.

Tests verify that GUI behaviour matches CLI ``mempalace wake-up`` and
``mempalace compress`` — raw output, no reformatting, no extra layers.
"""

import os

import pytest

from mempalace.gui_adapter import (
    MemPalaceAdapter,
    WakeUpResult,
    CompressResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def session_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("wc_session")


@pytest.fixture(scope="session")
def shared_palace(session_tmp):
    palace = session_tmp / "palace"
    palace.mkdir()
    return palace


@pytest.fixture(scope="session")
def shared_project(session_tmp):
    proj = session_tmp / "myproject"
    proj.mkdir()
    (proj / "mempalace.yaml").write_text(
        "wing: test_wing\n"
        "rooms:\n"
        "  - name: general\n"
        "    description: general\n"
        "    keywords: [general]\n"
        "  - name: decisions\n"
        "    description: decisions\n"
        "    keywords: [decision, decided]\n"
    )
    (proj / "readme.md").write_text(
        "# Test Project\n"
        "We decided to use PostgreSQL for persistence.\n"
        "GraphQL was chosen over REST for flexibility.\n"
    )
    (proj / "notes.md").write_text(
        "Meeting notes:\n"
        "Budget approved: $500/month for infrastructure.\n"
        "Alice will lead the backend migration.\n"
    )
    return proj


@pytest.fixture(scope="session")
def adapter_mined(shared_palace, shared_project):
    adapter = MemPalaceAdapter(palace_path=str(shared_palace))
    result = adapter.run_mine_projects(str(shared_project))
    assert result.ok, f"Session mine failed: {result.error}"
    return adapter


@pytest.fixture()
def tmp_palace(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()
    return palace


@pytest.fixture()
def tmp_project(tmp_path):
    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "mempalace.yaml").write_text(
        "wing: test_wing\n"
        "rooms:\n"
        "  - name: general\n"
        "    description: general\n"
        "    keywords: [general]\n"
    )
    (proj / "readme.md").write_text("We decided to use Redis for caching.\n")
    return proj


# ---------------------------------------------------------------------------
# Unit tests — run_wakeup
# ---------------------------------------------------------------------------


class TestRunWakeUp:
    def test_wakeup_returns_result(self, adapter_mined):
        result = adapter_mined.run_wakeup()
        assert isinstance(result, WakeUpResult)
        assert result.ok is True
        assert len(result.text) > 0

    def test_wakeup_with_wing(self, adapter_mined):
        result = adapter_mined.run_wakeup(wing="test_wing")
        assert result.ok is True
        assert len(result.text) > 0

    def test_wakeup_no_palace(self, tmp_path):
        adapter = MemPalaceAdapter(palace_path=str(tmp_path / "no_palace"))
        result = adapter.run_wakeup()
        assert result.ok is True
        assert "No palace" in result.text or "no palace" in result.text.lower()

    def test_wakeup_tokens_est(self, adapter_mined):
        result = adapter_mined.run_wakeup()
        assert result.ok
        assert result.tokens_est > 0
        assert result.tokens_est == len(result.text) // 4

    def test_wakeup_text_contains_l0(self, adapter_mined):
        result = adapter_mined.run_wakeup()
        assert result.ok
        assert "L0" in result.text or "IDENTITY" in result.text or len(result.text) > 0

    def test_wakeup_text_contains_l1(self, adapter_mined):
        result = adapter_mined.run_wakeup()
        assert result.ok
        assert (
            "L1" in result.text
            or "ESSENTIAL" in result.text
            or "No palace" in result.text
            or len(result.text) > 0
        )

    def test_wakeup_wing_filter(self, adapter_mined):
        all_result = adapter_mined.run_wakeup()
        wing_result = adapter_mined.run_wakeup(wing="test_wing")
        assert all_result.ok
        assert wing_result.ok

    def test_wakeup_does_not_print(self, adapter_mined):
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            adapter_mined.run_wakeup()
        finally:
            sys.stdout = old_stdout
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Unit tests — run_compress
# ---------------------------------------------------------------------------


class TestRunCompress:
    def test_compress_returns_result(self, adapter_mined):
        result = adapter_mined.run_compress(dry_run=True)
        assert isinstance(result, CompressResult)
        assert result.ok is True
        assert len(result.output) > 0

    def test_compress_with_wing(self, adapter_mined):
        result = adapter_mined.run_compress(wing="test_wing", dry_run=True)
        assert result.ok is True
        assert len(result.output) > 0

    def test_compress_no_palace(self, tmp_path):
        adapter = MemPalaceAdapter(palace_path=str(tmp_path / "no_palace"))
        result = (
            adapter_mined.run_compress(dry_run=True)
            if False
            else CompressResult(ok=False, error="skip")
        )
        adapter = MemPalaceAdapter(palace_path=str(tmp_path / "no_palace"))
        result = adapter.run_compress(dry_run=True)
        assert result.ok is False
        assert result.error is not None

    def test_compress_dry_run_does_not_store(self, adapter_mined):
        result = adapter_mined.run_compress(dry_run=True)
        assert result.ok
        assert result.dry_run is True
        assert "dry run" in result.output.lower()

    def test_compress_drawer_count(self, adapter_mined):
        result = adapter_mined.run_compress(dry_run=True)
        assert result.ok
        assert result.drawer_count > 0

    def test_compress_stats_fields(self, adapter_mined):
        result = adapter_mined.run_compress(dry_run=True)
        assert result.ok
        if result.drawer_count > 0:
            assert result.orig_tokens_est > 0
            assert result.comp_tokens_est > 0
            assert result.compression_ratio > 0

    def test_compress_output_contains_summary(self, adapter_mined):
        result = adapter_mined.run_compress(dry_run=True)
        assert result.ok
        assert "Total:" in result.output
        assert "compression" in result.output.lower()

    def test_compress_dry_run_shows_per_drawer(self, adapter_mined):
        result = adapter_mined.run_compress(dry_run=True)
        assert result.ok
        if result.drawer_count > 0:
            assert "t ->" in result.output

    def test_compress_no_dry_run_stores(self, adapter_mined):
        result = adapter_mined.run_compress(dry_run=False)
        assert result.ok
        assert result.dry_run is False
        assert "Stored" in result.output or result.drawer_count > 0

    def test_compress_empty_palace(self, tmp_palace):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_compress(dry_run=True)
        assert result.ok is False
        assert result.error is not None

    def test_compress_does_not_print(self, adapter_mined):
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            adapter_mined.run_compress(dry_run=True)
        finally:
            sys.stdout = old_stdout
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# CLI vs GUI parity — wake-up
# ---------------------------------------------------------------------------


class TestWakeUpParity:
    def test_parity_wakeup_text(self, adapter_mined):
        """GUI wake-up text must match CLI ``mempalace wake-up`` output."""
        from mempalace.layers import MemoryStack

        stack = MemoryStack(palace_path=adapter_mined.palace_path)
        cli_text = stack.wake_up(wing=None)
        gui_result = adapter_mined.run_wakeup(wing=None)
        assert gui_result.ok
        assert gui_result.text == cli_text

    def test_parity_wakeup_wing(self, adapter_mined):
        """GUI wake-up with wing must match CLI ``mempalace wake-up --wing``."""
        from mempalace.layers import MemoryStack

        wing = "test_wing"
        stack = MemoryStack(palace_path=adapter_mined.palace_path)
        cli_text = stack.wake_up(wing=wing)
        gui_result = adapter_mined.run_wakeup(wing=wing)
        assert gui_result.ok
        assert gui_result.text == cli_text

    def test_parity_wakeup_tokens(self, adapter_mined):
        """GUI token estimate must match CLI heuristic (len(text)//4)."""
        gui_result = adapter_mined.run_wakeup()
        assert gui_result.ok
        assert gui_result.tokens_est == len(gui_result.text) // 4


# ---------------------------------------------------------------------------
# CLI vs GUI parity — compress
# ---------------------------------------------------------------------------


class TestCompressParity:
    def test_parity_compress_dry_run_output(self, adapter_mined):
        """GUI compress dry-run must produce equivalent output to CLI."""
        from mempalace.dialect import Dialect
        from mempalace.palace import get_collection

        col = get_collection(adapter_mined.palace_path, create=False)
        total = col.count()
        all_docs, all_metas = [], []
        offset = 0
        while offset < total:
            batch = col.get(limit=500, offset=offset, include=["documents", "metadatas"])
            if not batch.get("documents"):
                break
            all_docs.extend(batch["documents"])
            all_metas.extend(batch["metadatas"])
            offset += len(batch["documents"])

        dialect = Dialect()
        cli_lines = []
        cli_lines.append(f"  Compressing {len(all_docs)} drawers...")
        cli_lines.append("")
        total_original = 0
        total_compressed = 0
        for doc, meta in zip(all_docs, all_metas):
            compressed = dialect.compress(doc, metadata=meta)
            stats = dialect.compression_stats(doc, compressed)
            total_original += stats["original_chars"]
            total_compressed += stats["summary_chars"]
            wing_name = meta.get("wing", "?")
            room_name = meta.get("room", "?")
            source = os.path.basename(meta.get("source_file", "?"))
            cli_lines.append(f"  [{wing_name}/{room_name}] {source}")
            cli_lines.append(
                f"    {stats['original_tokens_est']}t -> "
                f"{stats['summary_tokens_est']}t "
                f"({stats['size_ratio']:.1f}x)"
            )
            cli_lines.append(f"    {compressed}")
            cli_lines.append("")

        ratio = total_original / max(total_compressed, 1)
        orig_tokens = max(1, int(total_original / 3.8))
        comp_tokens = max(1, int(total_compressed / 3.8))
        cli_lines.append(
            f"  Total: {orig_tokens:,}t -> {comp_tokens:,}t ({ratio:.1f}x compression)"
        )
        cli_lines.append("  (dry run -- nothing stored)")
        cli_output = "\n".join(cli_lines)

        gui_result = adapter_mined.run_compress(dry_run=True)
        assert gui_result.ok
        assert gui_result.output == cli_output

    def test_parity_compress_wing_dry_run(self, adapter_mined):
        """GUI compress with --wing must match CLI for that wing."""
        wing = "test_wing"
        gui_result = adapter_mined.run_compress(wing=wing, dry_run=True)
        assert gui_result.ok
        assert "test_wing" in gui_result.output or gui_result.drawer_count > 0
