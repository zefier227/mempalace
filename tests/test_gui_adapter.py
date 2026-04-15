"""
tests/test_gui_adapter.py — Smoke tests for MemPalaceAdapter (gui_adapter.py)

Test plan covers:
  1. ChromaDB version check (arm64 safety gate)
  2. safe_init()       — non-interactive init
  3. run_mine_projects() — subprocess mine with progress streaming
  4. run_mine_convos()  — subprocess mine for chat exports
  5. run_search()       — direct Python API search
  6. run_status()       — structured status (no stdout)
  7. start/stop MCP server — subprocess lifecycle
  8. edge cases: missing palace, empty search, zero drawers

Performance notes
-----------------
The ``adapter_mined`` fixture is **session-scoped**: it mines the shared project
directory exactly once and all tests that depend on it reuse the same palace.
This keeps the full suite inside the 120-second pytest-timeout window.

Individual tests that need isolation (dry_run, nonexistent paths, etc.) create
their own temporary directories instead of using the shared fixture.

All tests use temporary directories; nothing written to the real palace.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from mempalace.gui_adapter import (
    MemPalaceAdapter,
    MineProgressEvent,
    SearchResult,
    PalaceStatus,
    InitResult,
    McpServerStatus,
)


# ---------------------------------------------------------------------------
# Session-level shared fixtures (mined once for the whole test run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def session_tmp(tmp_path_factory):
    """Persistent temp root shared across the whole test session."""
    return tmp_path_factory.mktemp("session")


@pytest.fixture(scope="session")
def shared_palace(session_tmp):
    """Single palace directory, created once per session."""
    palace = session_tmp / "palace"
    palace.mkdir()
    return palace


@pytest.fixture(scope="session")
def shared_project(session_tmp):
    """Small project directory with 3 text files and a mempalace.yaml."""
    proj = session_tmp / "myproject"
    proj.mkdir()
    (proj / "mempalace.yaml").write_text(
        "wing: smoke_test\n"
        "rooms:\n"
        "  - name: general\n"
        "    description: general notes\n"
        "    keywords: [general, notes]\n"
        "  - name: decisions\n"
        "    description: key decisions\n"
        "    keywords: [decision, decided, chose, switched]\n"
    )
    (proj / "README.md").write_text(
        "# My Project\n"
        "We switched to GraphQL because REST was too chatty.\n"
        "Decision: PostgreSQL for the main database.\n"
    )
    (proj / "notes.md").write_text(
        "Meeting notes 2026-04-01:\n"
        "Discussed pricing with the team. Budget approved: $500/month.\n"
        "Alice agreed to lead the backend migration.\n"
    )
    (proj / "architecture.md").write_text(
        "Architecture overview:\n"
        "- API layer: GraphQL with Apollo\n"
        "- Database: PostgreSQL 16\n"
        "- Cache: Redis\n"
    )
    return proj


@pytest.fixture(scope="session")
def adapter_mined(shared_palace, shared_project):
    """
    Session-scoped adapter with a pre-mined palace.

    Mines shared_project into shared_palace exactly once for the whole test
    session.  All tests that need a populated palace share this adapter.

    DO NOT call stop_mcp_server() from tests that use this fixture without
    pairing it with start_mcp_server(), as state leaks between tests.
    """
    adapter = MemPalaceAdapter(palace_path=str(shared_palace))
    result = adapter.run_mine_projects(str(shared_project))
    assert result.ok, f"Session mine failed: {result.error}"
    return adapter


# ---------------------------------------------------------------------------
# Function-level fixtures for tests that need isolation
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_palace(tmp_path):
    """Fresh isolated palace directory (function scope)."""
    palace = tmp_path / "palace"
    palace.mkdir()
    return palace


@pytest.fixture()
def tmp_project(tmp_path):
    """Small isolated project directory (function scope)."""
    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "mempalace.yaml").write_text(
        "wing: smoke_test\n"
        "rooms:\n"
        "  - name: general\n"
        "    description: general notes\n"
        "    keywords: [general, notes]\n"
        "  - name: decisions\n"
        "    description: key decisions\n"
        "    keywords: [decision, decided, chose, switched]\n"
    )
    (proj / "README.md").write_text(
        "# My Project\n"
        "We switched to GraphQL because REST was too chatty.\n"
        "Decision: PostgreSQL for the main database.\n"
    )
    (proj / "notes.md").write_text(
        "Meeting notes 2026-04-01:\n"
        "Discussed pricing with the team. Budget approved: $500/month.\n"
        "Alice agreed to lead the backend migration.\n"
    )
    (proj / "architecture.md").write_text(
        "Architecture overview:\n"
        "- API layer: GraphQL with Apollo\n"
        "- Database: PostgreSQL 16\n"
        "- Cache: Redis\n"
    )
    return proj


@pytest.fixture()
def tmp_convo(tmp_path):
    """Small conversation export directory (function scope)."""
    convo_dir = tmp_path / "conversations"
    convo_dir.mkdir()
    (convo_dir / "session1.md").write_text(
        "Human: Can you help me understand async/await in Python?\n"
        "Assistant: Sure! async/await is used for writing asynchronous code.\n"
        "It allows you to write non-blocking I/O without callback hell.\n\n"
        "Human: What about asyncio?\n"
        "Assistant: asyncio is the Python standard library that provides the event loop.\n"
    )
    return convo_dir


# ---------------------------------------------------------------------------
# 1. ChromaDB version check
# ---------------------------------------------------------------------------


class TestChromadbVersionCheck:
    def test_version_check_returns_dict(self):
        info = MemPalaceAdapter.check_chromadb_version()
        assert "version" in info
        assert "ok" in info
        assert "arm64_safe" in info
        assert "minimum" in info

    def test_version_is_safe(self):
        """Installed chromadb must be >= 1.5.4 for ARM64 safety."""
        info = MemPalaceAdapter.check_chromadb_version()
        assert info["ok"] is True, (
            f"ChromaDB {info['version']} is below minimum 1.5.4. "
            "macOS ARM64 segfaults are likely. "
            f"Warning: {info.get('warning')}"
        )
        assert info["arm64_safe"] is True

    def test_version_string_is_semver(self):
        info = MemPalaceAdapter.check_chromadb_version()
        if info["version"]:
            parts = info["version"].split(".")
            assert len(parts) >= 2
            assert all(p.isdigit() for p in parts[:2])


# ---------------------------------------------------------------------------
# 2. safe_init
# ---------------------------------------------------------------------------


class TestSafeInit:
    def test_basic_init_creates_config(self, tmp_path):
        """safe_init() should create config and return InitResult."""
        adapter = MemPalaceAdapter(palace_path=str(tmp_path / "palace"))
        result = adapter.safe_init()
        assert isinstance(result, InitResult)
        assert result.ok is True
        assert result.palace_path != ""

    def test_init_returns_palace_path(self, tmp_palace):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.safe_init()
        assert result.ok
        assert str(tmp_palace) in result.palace_path

    def test_init_with_room_detection(self, tmp_palace, tmp_project):
        """safe_init with auto_detect_rooms=True should not raise (--yes mode)."""
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.safe_init(
            project_dir=str(tmp_project),
            auto_detect_rooms=True,
        )
        assert result.ok

    def test_init_idempotent(self, tmp_palace):
        """Calling safe_init twice should not fail."""
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        r1 = adapter.safe_init()
        r2 = adapter.safe_init()
        assert r1.ok
        assert r2.ok

    def test_palace_path_property(self, tmp_palace):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        assert adapter.palace_path == str(tmp_palace)


# ---------------------------------------------------------------------------
# 3. run_mine_projects
# ---------------------------------------------------------------------------


class TestRunMineProjects:
    def test_mine_returns_ok(self, tmp_palace, tmp_project):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_mine_projects(str(tmp_project))
        assert result.ok is True, f"Mine failed: {result.error}"

    def test_mine_counts_drawers(self, tmp_palace, tmp_project):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_mine_projects(str(tmp_project))
        assert result.ok
        # 3 files in tmp_project → at least 1 drawer each
        assert result.drawers_filed >= 1

    def test_mine_records_wing(self, tmp_palace, tmp_project):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_mine_projects(str(tmp_project))
        assert result.ok
        assert result.wing == "smoke_test"  # from mempalace.yaml

    def test_mine_progress_callback(self, tmp_palace, tmp_project):
        """Progress callback must receive file events and at least one done event."""
        events = []
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        adapter.run_mine_projects(str(tmp_project), on_progress=events.append)

        file_events = [e for e in events if e.type == "file"]
        done_events = [e for e in events if e.type == "done"]

        assert len(file_events) >= 1, "Expected at least one file progress event"
        assert len(done_events) == 1, "Expected exactly one done event"
        assert done_events[0].drawers_filed >= 1

    def test_mine_dry_run(self, tmp_palace, tmp_project):
        """Dry-run should not create any drawers in the palace."""
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_mine_projects(str(tmp_project), dry_run=True)
        assert result.ok
        # After dry run, palace should have zero drawers
        status = adapter.run_status()
        # No palace created in dry run (FileNotFoundError expected or 0 drawers)
        if status.ok:
            assert status.total_drawers == 0

    def test_mine_with_wing_override(self, tmp_palace, tmp_project):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_mine_projects(str(tmp_project), wing="custom_wing")
        assert result.ok
        assert result.wing == "custom_wing"

    def test_mine_nonexistent_dir_returns_zero_files(self, tmp_palace):
        """MemPalace mine on a nonexistent dir exits 0 and processes 0 files.

        miner.scan_project() returns an empty list for missing dirs (os.walk
        simply yields nothing), so this is considered a successful no-op
        rather than an error at the CLI level. The adapter reflects that.
        """
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_mine_projects("/nonexistent/path/that/does/not/exist")
        # No crash, returns ok (0 files is a valid outcome from upstream)
        assert result.error is None or result.ok is True
        assert result.files_processed == 0

    def test_mine_idempotent(self, tmp_palace, tmp_project):
        """Mining twice should not raise; files are skipped on second pass."""
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        r1 = adapter.run_mine_projects(str(tmp_project))
        r2 = adapter.run_mine_projects(str(tmp_project))
        assert r1.ok
        assert r2.ok
        # Second run: all files already filed → drawers_filed=0, files_skipped>0
        assert r2.files_skipped >= 1 or r2.drawers_filed == 0


# ---------------------------------------------------------------------------
# 4. run_mine_convos
# ---------------------------------------------------------------------------


class TestRunMineConvos:
    def test_mine_convos_ok(self, tmp_palace, tmp_convo):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_mine_convos(str(tmp_convo))
        assert result.ok is True, f"Mine convos failed: {result.error}"

    def test_mine_convos_drawers(self, tmp_palace, tmp_convo):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_mine_convos(str(tmp_convo))
        assert result.ok
        assert result.drawers_filed >= 1


# ---------------------------------------------------------------------------
# 5. run_search
# ---------------------------------------------------------------------------


class TestRunSearch:
    def test_search_returns_structured_result(self, adapter_mined):
        result = adapter_mined.run_search("GraphQL")
        assert isinstance(result, SearchResult)
        assert result.ok is True
        assert result.query == "GraphQL"

    def test_search_finds_known_content(self, adapter_mined):
        result = adapter_mined.run_search("GraphQL REST API decision")
        assert result.ok
        assert len(result.hits) >= 1
        # Top hit should contain the GraphQL mention
        top = result.hits[0]
        assert "GraphQL" in top.text or top.similarity > 0.3

    def test_search_result_has_all_fields(self, adapter_mined):
        result = adapter_mined.run_search("PostgreSQL database")
        assert result.ok
        assert len(result.hits) >= 1
        hit = result.hits[0]
        assert isinstance(hit.text, str) and len(hit.text) > 0
        assert isinstance(hit.wing, str)
        assert isinstance(hit.room, str)
        assert isinstance(hit.source_file, str)
        assert 0.0 <= hit.similarity <= 1.0
        assert isinstance(hit.distance, float)

    def test_search_no_results_returns_empty(self, adapter_mined):
        """Searching for something totally unrelated should return ok=True, hits=[]."""
        result = adapter_mined.run_search(
            "xkcd_unique_string_that_cannot_match_anything_1234567",
            max_distance=0.01,  # very strict — forces no results
        )
        assert result.ok is True
        assert len(result.hits) == 0

    def test_search_wing_filter(self, adapter_mined):
        result = adapter_mined.run_search(
            "architecture database", wing="smoke_test"
        )
        assert result.ok
        for hit in result.hits:
            assert hit.wing == "smoke_test"

    def test_search_nonexistent_palace(self, tmp_path):
        adapter = MemPalaceAdapter(palace_path=str(tmp_path / "no_palace_here"))
        result = adapter.run_search("anything")
        assert result.ok is False
        assert result.error is not None

    def test_search_n_results_respected(self, adapter_mined):
        result = adapter_mined.run_search("project", n_results=1)
        assert result.ok
        assert len(result.hits) <= 1

    def test_search_total_candidates_field(self, adapter_mined):
        result = adapter_mined.run_search("Python")
        assert isinstance(result.total_candidates, int)
        assert result.total_candidates >= 0


# ---------------------------------------------------------------------------
# 6. run_status
# ---------------------------------------------------------------------------


class TestRunStatus:
    def test_status_no_palace_returns_error(self, tmp_path):
        adapter = MemPalaceAdapter(palace_path=str(tmp_path / "empty"))
        status = adapter.run_status()
        assert isinstance(status, PalaceStatus)
        assert status.ok is False
        assert status.error is not None

    def test_status_after_mine(self, adapter_mined):
        status = adapter_mined.run_status()
        assert status.ok is True
        assert status.total_drawers >= 1
        assert len(status.wings) >= 1
        assert status.palace_path != ""
        assert status.chromadb_version != ""

    def test_status_wing_structure(self, adapter_mined):
        status = adapter_mined.run_status()
        assert status.ok
        wing = status.wings[0]
        assert wing.name != ""
        assert len(wing.rooms) >= 1
        assert wing.total_drawers >= 1

    def test_status_room_count_accurate(self, adapter_mined):
        status = adapter_mined.run_status()
        assert status.ok
        total_from_rooms = sum(
            room.drawers
            for wing in status.wings
            for room in wing.rooms
        )
        assert total_from_rooms == status.total_drawers

    def test_status_chromadb_version_in_result(self, adapter_mined):
        status = adapter_mined.run_status()
        assert status.ok
        assert "." in status.chromadb_version  # looks like a version string

    def test_status_does_not_print(self, adapter_mined):
        """run_status must not print anything to stdout.

        We use a manual stdout capture instead of capsys because the
        adapter_mined fixture is session-scoped (capsys requires function scope).
        """
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            adapter_mined.run_status()
        finally:
            sys.stdout = old_stdout
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# 7. MCP server lifecycle
# ---------------------------------------------------------------------------


class TestMcpServer:
    def test_start_returns_status(self, adapter_mined):
        status = adapter_mined.start_mcp_server()
        assert isinstance(status, McpServerStatus)
        try:
            if status.running:
                assert status.pid is not None
                assert status.pid > 0
        finally:
            adapter_mined.stop_mcp_server()

    def test_start_stop_cycle(self, adapter_mined):
        start = adapter_mined.start_mcp_server()
        try:
            assert start.running is True
            server_status = adapter_mined.mcp_server_status()
            assert server_status.running is True
        finally:
            stop = adapter_mined.stop_mcp_server()
        assert stop.running is False

    def test_stop_when_not_running(self, tmp_palace):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        # Should not raise even if server was never started
        result = adapter.stop_mcp_server()
        assert result.running is False

    def test_double_start_returns_same_pid(self, adapter_mined):
        s1 = adapter_mined.start_mcp_server()
        s2 = adapter_mined.start_mcp_server()
        try:
            assert s1.running
            assert s2.running
            assert s1.pid == s2.pid  # second call returns existing server
        finally:
            adapter_mined.stop_mcp_server()

    def test_mcp_responds_to_ping(self, adapter_mined):
        """MCP server must respond to JSON-RPC ping.

        start_mcp_server() reads the initialize response during startup.
        The ping response is therefore the NEXT line on stdout.
        We loop with a 5-second deadline to tolerate any startup lag.
        """
        import select

        start = adapter_mined.start_mcp_server()
        assert start.running, f"Server failed to start: {start.error}"

        proc = adapter_mined._mcp_proc
        try:
            ping = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping", "params": {}})
            proc.stdin.write(ping + "\n")
            proc.stdin.flush()

            # Read lines until we get id=99 (skip any buffered responses)
            deadline = time.time() + 5.0
            resp = None
            while time.time() < deadline:
                ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                if not ready:
                    continue
                line = proc.stdout.readline()
                if not line:
                    break
                candidate = json.loads(line)
                if candidate.get("id") == 99:
                    resp = candidate
                    break

            assert resp is not None, "MCP server did not return ping response within 5s"
            assert resp.get("id") == 99
            assert "result" in resp
        finally:
            adapter_mined.stop_mcp_server()


# ---------------------------------------------------------------------------
# 8. check_palace_exists utility
# ---------------------------------------------------------------------------


class TestCheckPalaceExists:
    def test_nonexistent_palace(self, tmp_path):
        info = MemPalaceAdapter.check_palace_exists(str(tmp_path / "nope"))
        assert info["exists"] is False
        assert info["has_db"] is False

    def test_empty_dir_no_db(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        info = MemPalaceAdapter.check_palace_exists(str(d))
        assert info["exists"] is True
        assert info["has_db"] is False

    def test_mined_palace_has_db(self, tmp_palace, tmp_project):
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        adapter.run_mine_projects(str(tmp_project))
        info = MemPalaceAdapter.check_palace_exists(str(tmp_palace))
        assert info["exists"] is True
        assert info["has_db"] is True


# ---------------------------------------------------------------------------
# 9. Thread safety (serial search validation)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_serial_searches_all_succeed(self, adapter_mined):
        """Serial searches (one at a time) always succeed.

        ChromaDB 1.5.x uses a Rust backend with file-level locking —
        concurrent PersistentClient instances to the same path can fail
        with 'Could not connect to tenant'. The adapter (and therefore
        the GUI) should issue searches serially from a single thread.
        This test verifies the serial path is fully reliable.
        """
        queries = ["GraphQL", "database", "architecture", "decisions", "budget"]
        results = [adapter_mined.run_search(q, n_results=2) for q in queries]

        assert all(r.ok for r in results), [
            f"{q}: {r.error}" for q, r in zip(queries, results) if not r.ok
        ]
        assert len(results) == 5
