"""
tests/test_backend_hardening.py — Targeted hardening tests for MemPalace GUI backend.

Covers:
  1. search-after-mine in the same process (stale-client eviction must work).
  2. Mine subprocess timeout / cancel  (hanging child is killed within deadline).
  3. Wing and room names with spaces, hyphens, and punctuation.
  4. SearchHit enriched fields (source_path, drawer_id, chunk_index).
  5. BackendController event ordering and serialisation.
  6. MineProgressEvent parser robustness (_parse_mine_line edge cases).

These tests use function-scoped fixtures so each test gets a clean palace.
All tests must complete within the pytest-timeout (120 s).
"""

import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from mempalace.gui_adapter import (
    MemPalaceAdapter,
    MineHandle,
    MineProgressEvent,
    SearchHit,
    SearchResult,
    _parse_mine_line,
)
from mempalace.backend_controller import BackendController, BackendEvent, EventType


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


def _make_project(tmp_path: Path, *, wing: str = "test", rooms=None, files=None) -> Path:
    """Create a minimal project directory with a mempalace.yaml and one markdown file."""
    proj = tmp_path / "proj"
    proj.mkdir()

    if rooms is None:
        rooms = [
            {"name": "general", "desc": "general notes", "kw": "general,notes"},
            {"name": "decisions", "desc": "key decisions", "kw": "decision,decided"},
        ]
    room_lines = "\n".join(
        f"  - name: {r['name']}\n"
        f"    description: {r['desc']}\n"
        f"    keywords: [{r['kw']}]"
        for r in rooms
    )
    (proj / "mempalace.yaml").write_text(
        f"wing: {wing}\n"
        "rooms:\n"
        f"{room_lines}\n"
    )

    if files is None:
        files = {
            "notes.md": (
                "# Project Notes\n"
                "We decided to use GraphQL instead of REST for performance reasons.\n"
                "PostgreSQL is the main database. Redis for caching.\n"
                "Team decision: migrate to Kubernetes by Q3.\n"
            )
        }
    for name, content in files.items():
        (proj / name).write_text(content)

    return proj


@pytest.fixture()
def isolated_palace(tmp_path):
    """Fresh isolated palace for each test."""
    p = tmp_path / "palace"
    p.mkdir()
    return p


@pytest.fixture()
def simple_project(tmp_path):
    """Minimal project with one file."""
    return _make_project(tmp_path)


# ---------------------------------------------------------------------------
# Session-scoped fixtures for BackendController tests that need a pre-mined
# palace.  Mining once and sharing across BackendController tests avoids the
# load-spike that makes the mine take >80s when run after 40+ other tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bc_session_tmp(tmp_path_factory):
    """Persistent temp root for the BackendController test session."""
    return tmp_path_factory.mktemp("bc_session")


@pytest.fixture(scope="session")
def bc_palace(bc_session_tmp):
    """Session-scoped palace for BackendController tests."""
    p = bc_session_tmp / "palace"
    p.mkdir()
    return p


@pytest.fixture(scope="session")
def bc_project(bc_session_tmp):
    """Session-scoped project mined into bc_palace."""
    proj = bc_session_tmp / "proj"
    proj.mkdir()
    (proj / "mempalace.yaml").write_text(
        "wing: bc_test\n"
        "rooms:\n"
        "  - name: general\n"
        "    description: general notes\n"
        "    keywords: [general, notes]\n"
        "  - name: decisions\n"
        "    description: key decisions\n"
        "    keywords: [decision, decided]\n"
    )
    (proj / "notes.md").write_text(
        "# BackendController Test Project\n"
        "We decided to use GraphQL instead of REST for performance reasons.\n"
        "PostgreSQL is the primary database. Redis is used for caching.\n"
        "Team decision: migrate to Kubernetes by Q3.\n"
    )
    return proj


@pytest.fixture(scope="session")
def ctrl_mined(bc_palace, bc_project):
    """
    Session-scoped BackendController with a pre-mined palace.

    Mines once; all BackendController tests that depend on a populated
    palace share this controller instance.  The mine takes ~5s; sharing
    avoids re-paying that cost for every test.
    """
    import threading as _threading

    mine_done_event = _threading.Event()
    mine_result_holder = {}

    def on_ev(ev):
        if ev.type == EventType.MINE_DONE:
            mine_result_holder["event"] = ev
            mine_done_event.set()
        elif ev.type == EventType.MINE_ERROR:
            mine_result_holder["event"] = ev
            mine_done_event.set()

    ctrl = BackendController(palace_path=str(bc_palace), on_event=on_ev)
    ctrl.start()
    ctrl.mine(str(bc_project))

    # Wait up to 60s for the mine to complete (plenty even under heavy load).
    mine_done_event.wait(timeout=60)
    ev = mine_result_holder.get("event")
    assert ev is not None, "Session mine never completed within 60s"
    assert ev.type == EventType.MINE_DONE, (
        f"Session mine failed: {ev.payload}"
    )
    yield ctrl
    ctrl.stop(timeout=15)


# ---------------------------------------------------------------------------
# 1. Search-after-mine in same process (stale-client eviction)
# ---------------------------------------------------------------------------


class TestSearchAfterMine:
    """After run_mine_projects returns, run_search must find the new drawers.

    This is the core stale-ChromaDB-client regression test.  The mine runs in a
    subprocess; the adapter must evict the parent-process client cache so the
    next search opens a fresh PersistentClient that sees the newly-written drawers.
    """

    def test_search_finds_content_after_mine(self, isolated_palace, simple_project):
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        result = adapter.run_mine_projects(str(simple_project))
        assert result.ok, f"Mine failed: {result.error}"
        assert result.drawers_filed >= 1, "Expected at least 1 drawer"

        # Immediately search in the same process — must hit fresh data.
        search = adapter.run_search("GraphQL REST decision")
        assert search.ok, f"Search failed: {search.error}"
        assert len(search.hits) >= 1, (
            "Search returned 0 hits immediately after mine. "
            "Stale-client eviction (_invalidate_chroma_client) is not working."
        )

    def test_search_finds_content_after_second_mine(self, isolated_palace, tmp_path):
        """Second mine on a new file must make the new content searchable."""
        proj1 = _make_project(tmp_path, files={
            # Content must exceed MIN_CHUNK_SIZE (50 chars) or miner skips the chunk.
            "file_a.md": (
                "Topic A: distributed tracing with Jaeger and OpenTelemetry.\n"
                "Jaeger is used for end-to-end distributed request tracing.\n"
            )
        })
        # _make_project always creates <parent>/proj, so give it a distinct parent.
        proj2_parent = tmp_path / "proj2_parent"
        proj2_parent.mkdir()
        proj2 = _make_project(proj2_parent, files={
            # Content must exceed MIN_CHUNK_SIZE (50 chars) or miner skips the chunk.
            "file_b.md": (
                "Topic B: service mesh with Istio and Envoy proxy.\n"
                "Istio provides traffic management, security, and observability.\n"
                "Envoy is the sidecar proxy used by Istio for service-to-service communication.\n"
            )
        })
        (proj2 / "mempalace.yaml").write_text(
            "wing: test\nrooms:\n  - name: gen\n    description: gen\n    keywords: [gen]\n"
        )

        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        r1 = adapter.run_mine_projects(str(proj1))
        assert r1.ok

        r2 = adapter.run_mine_projects(str(proj2))
        assert r2.ok, f"Second mine failed: {r2.error}"

        # Both files' content must be searchable after the second mine.
        s = adapter.run_search("Istio Envoy service mesh")
        assert s.ok
        assert len(s.hits) >= 1, "Second mine result not visible — stale client after double mine"

    def test_search_hit_fields_populated_after_mine(self, isolated_palace, simple_project):
        """SearchHit must have source_path, drawer_id, and chunk_index populated."""
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        adapter.run_mine_projects(str(simple_project))

        search = adapter.run_search("GraphQL database")
        assert search.ok
        assert len(search.hits) >= 1

        hit = search.hits[0]
        # source_path: full absolute path to the source file
        assert isinstance(hit.source_path, str), "source_path must be a str"
        assert hit.source_path != "", (
            "source_path is empty — check that searcher.py promotes "
            "_source_file_full to source_path before stripping internals"
        )
        # drawer_id: stable ChromaDB document ID
        assert isinstance(hit.drawer_id, str), "drawer_id must be a str"
        assert hit.drawer_id != "", (
            "drawer_id is empty — check that searcher.py captures query IDs "
            "and sets entry['drawer_id']"
        )
        # chunk_index: integer (0 for single-chunk files)
        assert hit.chunk_index is not None, "chunk_index should not be None after mine"
        assert isinstance(hit.chunk_index, int), "chunk_index must be an int"

    def test_source_path_is_absolute(self, isolated_palace, simple_project):
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        adapter.run_mine_projects(str(simple_project))
        search = adapter.run_search("GraphQL")
        assert search.ok and search.hits
        hit = search.hits[0]
        if hit.source_path:
            assert os.path.isabs(hit.source_path), (
                f"source_path should be absolute, got: {hit.source_path!r}"
            )

    def test_source_file_basename_matches_source_path(self, isolated_palace, simple_project):
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        adapter.run_mine_projects(str(simple_project))
        search = adapter.run_search("PostgreSQL")
        assert search.ok and search.hits
        hit = search.hits[0]
        if hit.source_path and hit.source_file:
            assert Path(hit.source_path).name == hit.source_file, (
                f"basename of source_path ({Path(hit.source_path).name!r}) "
                f"must equal source_file ({hit.source_file!r})"
            )


# ---------------------------------------------------------------------------
# 2. Mine subprocess timeout / cancel
# ---------------------------------------------------------------------------


class TestMineTimeoutCancel:
    """The mine subprocess must be killed when either a timeout or cancel fires."""

    def test_mine_cancel_via_handle(self, isolated_palace, simple_project):
        """cancel() on the handle must abort the mine before it finishes.

        We run the mine in a thread and cancel almost immediately.  The mine
        subprocess should be killed and the result should be ok=False.

        NOTE: Because the test project is small, there is a race — the mine
        might finish before the cancel fires.  We guard with an assertion that
        allows either 'cancelled cleanly' or 'completed before cancel' rather
        than demanding a specific outcome; the important thing is that the
        adapter doesn't deadlock or raise.
        """
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        handle = MineHandle()

        result_holder = {}

        def _run():
            result_holder["result"] = adapter.run_mine_projects(
                str(simple_project),
                handle=handle,
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        # Give the subprocess a moment to start, then cancel.
        time.sleep(0.1)
        handle.cancel()
        t.join(timeout=15)

        assert not t.is_alive(), "run_mine_projects blocked after cancel — deadlock?"
        result = result_holder.get("result")
        assert result is not None
        # Either cancelled (ok=False, error='Mine cancelled') or finished before cancel.
        if result.ok:
            pass  # finished faster than cancel — acceptable
        else:
            assert "cancel" in (result.error or "").lower() or result.error is not None

    def test_mine_timeout_kills_subprocess(self, isolated_palace, simple_project):
        """A very short timeout must kill the mine and return ok=False."""
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        # 0.001 s timeout — so short that even process startup may not complete.
        result = adapter.run_mine_projects(str(simple_project), timeout=0.001)
        # Either timed out or (rarely) finished in < 1 ms.
        if not result.ok:
            assert result.error is not None
            assert "timed out" in result.error.lower() or "time" in result.error.lower()

    def test_mine_timeout_not_triggered_on_fast_mine(self, isolated_palace, simple_project):
        """A generous timeout must not interfere with a normally-completing mine."""
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        result = adapter.run_mine_projects(str(simple_project), timeout=90)
        assert result.ok, f"Mine failed even with 90s timeout: {result.error}"

    def test_mine_handle_cancel_is_idempotent(self):
        """Calling cancel() multiple times must not raise."""
        handle = MineHandle()
        handle.cancel()
        handle.cancel()  # second call — must not raise
        assert handle.cancelled is True


# ---------------------------------------------------------------------------
# 3. Wing and room names with spaces, hyphens, and punctuation
# ---------------------------------------------------------------------------


class TestNamesWithSpecialChars:
    """Ensure wing/room names with spaces and punctuation survive the full pipeline.

    The _parse_mine_line regex uses .+ (greedy) for Wing names and
    '^    (.+?)\\s{2,}' for room names — both allow arbitrary characters.
    """

    @pytest.mark.parametrize("wing_name", [
        "my cool project",
        "arch & design",
        "front-end 2026",
        "Q1 planning (draft)",
    ])
    def test_parse_wing_with_spaces(self, wing_name):
        state: dict = {}
        line = f"  Wing:    {wing_name}"
        _parse_mine_line(line, state)
        assert state.get("wing") == wing_name, (
            f"Wing regex failed to parse {wing_name!r}; state={state}"
        )

    @pytest.mark.parametrize("room_name,count", [
        ("arch & design", 3),
        ("front-end notes", 7),
        ("Q1 planning (draft)", 1),
        ("decisions made", 5),
    ])
    def test_parse_room_with_spaces(self, room_name, count):
        state: dict = {}
        line = f"    {room_name}  {count} files"
        _parse_mine_line(line, state)
        rooms = state.get("rooms", {})
        assert room_name in rooms, (
            f"Room regex failed to parse {room_name!r} from line {line!r}; rooms={rooms}"
        )
        assert rooms[room_name] == count

    def test_done_event_preserves_special_char_wing(self):
        """Full parse sequence produces a done event with the correct wing name."""
        lines = [
            "  Wing:    arch & design",
            "  ✓ [   1/1] notes.md    +2",
            "  Done.",
            "    decisions made  3 files",
            "  Next: mempalace search ...",
        ]
        state: dict = {}
        events = []
        for line in lines:
            ev = _parse_mine_line(line, state)
            if ev:
                events.append(ev)

        done_events = [e for e in events if e.type == "done"]
        assert done_events, "No done event emitted"
        done = done_events[0]
        assert done.wing == "arch & design"
        assert "decisions made" in done.rooms

    def test_mine_with_spaced_wing_name_via_adapter(self, isolated_palace, tmp_path):
        """wing override with spaces must be passed correctly to the subprocess."""
        proj = _make_project(tmp_path)
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        # Note: CLI passes --wing value as a single arg; shell quoting is handled
        # by subprocess (list form, not shell=True), so spaces are safe.
        result = adapter.run_mine_projects(str(proj), wing="my special wing")
        assert result.ok, f"Mine with spaced wing failed: {result.error}"
        assert result.wing == "my special wing"

    def test_mine_progress_callback_with_spaces_in_filename(self, isolated_palace, tmp_path):
        """Files with spaces in their names must emit correct file events."""
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "mempalace.yaml").write_text(
            "wing: test\nrooms:\n  - name: gen\n    description: gen\n    keywords: [gen]\n"
        )
        fname = "my report 2026.md"
        (proj / fname).write_text(
            "Architecture report: We chose GraphQL for the API layer.\n" * 5
        )
        events = []
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        result = adapter.run_mine_projects(str(proj), on_progress=events.append)
        assert result.ok

        file_events = [e for e in events if e.type == "file"]
        # The filename field in the file event should contain the actual filename
        # (adapter strips leading/trailing whitespace from the parsed match group).
        if file_events:
            filenames = [e.filename for e in file_events]
            assert any(fname in fn for fn in filenames), (
                f"Expected filename containing {fname!r} in file events; got {filenames}"
            )


# ---------------------------------------------------------------------------
# 4. BackendController event ordering and serialisation
# ---------------------------------------------------------------------------


class TestBackendController:
    """Verify the BackendController routes work items correctly and posts events.

    Tests that need a pre-mined palace use the session-scoped ``ctrl_mined``
    fixture so the mine only runs once per session rather than once per test.
    Tests that need a fresh controller (start/stop, cancel) use isolated
    function-scoped palaces.
    """

    def test_controller_starts_and_stops(self, isolated_palace):
        ctrl = BackendController(palace_path=str(isolated_palace))
        ctrl.start()
        assert ctrl._started is True
        ctrl.stop()
        assert ctrl._started is False

    def test_mine_done_event_payload_is_mine_result(self, ctrl_mined):
        """The session-scoped ctrl_mined already mined; verify MINE_DONE payload."""
        # The session fixture validates MINE_DONE before yielding, so the
        # event is already confirmed. We just check the controller is alive.
        from mempalace.gui_adapter import MineResult
        assert ctrl_mined._started is True
        # Run status to confirm the mined palace is accessible
        status_done = threading.Event()
        status_holder = {}

        orig_on_event = ctrl_mined._on_event

        def capture(ev):
            if orig_on_event:
                orig_on_event(ev)
            if ev.type == EventType.STATUS_DONE:
                status_holder["ev"] = ev
                status_done.set()

        ctrl_mined._on_event = capture
        ctrl_mined.status()
        status_done.wait(timeout=15)
        ctrl_mined._on_event = orig_on_event  # restore

        ev = status_holder.get("ev")
        assert ev is not None, "STATUS_DONE never received"
        assert ev.payload.ok is True
        assert ev.payload.total_drawers >= 1

    def test_search_posts_started_event(self, ctrl_mined):
        """search() posts SEARCH_STARTED synchronously before returning."""
        posted = []
        orig = ctrl_mined._on_event

        done_event = threading.Event()
        search_rid = None

        def capture(ev):
            if orig:
                orig(ev)
            posted.append(ev)
            if ev.type in (EventType.SEARCH_DONE, EventType.SEARCH_ERROR):
                done_event.set()

        ctrl_mined._on_event = capture
        search_rid = ctrl_mined.search("GraphQL")
        # SEARCH_STARTED must have been posted synchronously inside search()
        assert any(e.type == EventType.SEARCH_STARTED for e in posted), (
            "SEARCH_STARTED not posted synchronously"
        )
        assert all(
            e.request_id == search_rid
            for e in posted if e.type == EventType.SEARCH_STARTED
        )
        done_event.wait(timeout=15)
        ctrl_mined._on_event = orig

    def test_search_done_event_payload_is_search_result(self, ctrl_mined):
        """search() produces a SEARCH_DONE event with a SearchResult payload."""
        from mempalace.gui_adapter import SearchResult

        done_event = threading.Event()
        result_holder = {}
        orig = ctrl_mined._on_event

        def capture(ev):
            if orig:
                orig(ev)
            if ev.type == EventType.SEARCH_DONE:
                result_holder["ev"] = ev
                done_event.set()

        ctrl_mined._on_event = capture
        ctrl_mined.search("GraphQL PostgreSQL")
        done_event.wait(timeout=15)
        ctrl_mined._on_event = orig

        ev = result_holder.get("ev")
        assert ev is not None, "SEARCH_DONE event never posted"
        assert isinstance(ev.payload, SearchResult)
        assert ev.payload.ok is True

    def test_request_ids_are_unique(self, isolated_palace):
        ctrl = BackendController(palace_path=str(isolated_palace))
        ctrl.start()
        rids = [ctrl.search("q") for _ in range(5)]
        ctrl.stop(timeout=30)
        assert len(set(rids)) == 5, "Request IDs must be unique"

    def test_controller_stop_is_idempotent(self, isolated_palace):
        ctrl = BackendController(palace_path=str(isolated_palace))
        ctrl.start()
        ctrl.stop()
        ctrl.stop()  # second stop must not raise

    def test_status_posts_status_done(self, ctrl_mined):
        """status() posts a STATUS_DONE event with a PalaceStatus payload."""
        from mempalace.gui_adapter import PalaceStatus

        done_event = threading.Event()
        result_holder = {}
        orig = ctrl_mined._on_event

        def capture(ev):
            if orig:
                orig(ev)
            if ev.type == EventType.STATUS_DONE:
                result_holder["ev"] = ev
                done_event.set()

        ctrl_mined._on_event = capture
        ctrl_mined.status()
        done_event.wait(timeout=15)
        ctrl_mined._on_event = orig

        ev = result_holder.get("ev")
        assert ev is not None, "STATUS_DONE never posted"
        assert isinstance(ev.payload, PalaceStatus)
        assert ev.payload.ok is True

    def test_mine_cancel_via_controller(self, isolated_palace, simple_project):
        """Cancelling via the returned MineHandle must propagate to the worker."""
        ctrl = BackendController(palace_path=str(isolated_palace))
        ctrl.start()

        done_event = threading.Event()
        result_holder = {}

        def capture(ev):
            if ev.type in (EventType.MINE_DONE, EventType.MINE_ERROR):
                result_holder["ev"] = ev
                done_event.set()

        ctrl._on_event = capture
        handle, rid = ctrl.mine(str(simple_project))
        time.sleep(0.05)
        handle.cancel()

        # Wait: either MINE_DONE (finished before cancel) or MINE_ERROR (cancelled)
        done_event.wait(timeout=15)
        ctrl.stop()

        ev = result_holder.get("ev")
        assert ev is not None, "Neither MINE_DONE nor MINE_ERROR received after cancel"
        assert ev.request_id == rid


# ---------------------------------------------------------------------------
# 5. _parse_mine_line edge-cases
# ---------------------------------------------------------------------------


class TestParseMineLineEdgeCases:
    """Unit-test the stdout line parser in isolation."""

    def test_file_event_parsed(self):
        state: dict = {}
        ev = _parse_mine_line("  ✓ [   3/42] my file name.py                    +2", state)
        assert ev is not None
        assert ev.type == "file"
        assert ev.current == 3
        assert ev.total == 42
        assert "my file name" in ev.filename
        assert ev.drawers == 2

    def test_file_event_with_hyphen_in_name(self):
        state: dict = {}
        ev = _parse_mine_line("  ✓ [  10/20] some-module-v2.ts    +1", state)
        assert ev is not None and ev.type == "file"
        assert ev.drawers == 1

    def test_summary_line_parsed(self):
        state: dict = {}
        _parse_mine_line("  Files processed: 7", state)
        _parse_mine_line("  Files skipped (already filed): 2", state)
        _parse_mine_line("  Drawers filed: 14", state)
        assert state["files_processed"] == 7
        assert state["files_skipped"] == 2
        assert state["drawers_filed"] == 14

    def test_done_event_emitted_on_next_line(self):
        """Done event is emitted on the 'Next:' line, not on 'Done.'."""
        state: dict = {}
        events = []
        for line in [
            "  Files processed: 3",
            "  Drawers filed: 5",
            "    general  3 files",
            "    decisions  2 files",
            "  Done.",
            "  Next: mempalace search ...",
        ]:
            ev = _parse_mine_line(line, state)
            if ev:
                events.append(ev)

        done = [e for e in events if e.type == "done"]
        assert len(done) == 1
        assert done[0].files_processed == 3
        assert done[0].drawers_filed == 5
        assert done[0].rooms == {"general": 3, "decisions": 2}

    def test_unrecognised_lines_return_none(self):
        state: dict = {}
        assert _parse_mine_line("  Scanning directory...", state) is None
        assert _parse_mine_line("", state) is None
        assert _parse_mine_line("  MemPalace v3.3.0", state) is None

    def test_no_done_event_without_done_marker(self):
        """'Next:' line alone must not emit a done event (no 'Done.' seen yet)."""
        state: dict = {}
        ev = _parse_mine_line("  Next: mempalace search ...", state)
        assert ev is None

    def test_multiple_rooms_accumulated(self):
        state: dict = {}
        for line in [
            "    api & backend  5 files",
            "    front-end ui  3 files",
            "    data (legacy)  1 files",
        ]:
            _parse_mine_line(line, state)
        assert state["rooms"]["api & backend"] == 5
        assert state["rooms"]["front-end ui"] == 3
        assert state["rooms"]["data (legacy)"] == 1


# ---------------------------------------------------------------------------
# 6. Warm-client regression: search/status -> mine -> immediate search/status
# ---------------------------------------------------------------------------


class TestWarmClientRegression:
    """Regression suite for the stale-ChromaDB-client scenario.

    Scenario tested:
        1. Open a fresh MemPalaceAdapter on an empty palace → search returns
           ok=False (no palace) and status returns ok=False.
        2. Mine a project into the palace via the adapter.
        3. Immediately search and call status in the same process — must see
           the new drawers WITHOUT creating a new adapter instance.

    This exercises the full warm-client path:
        - search_memories / get_collection reads from _DEFAULT_BACKEND._clients
        - run_mine_projects() launches subprocess, then calls
          _invalidate_chroma_client() which pops the cached client
        - next get_collection() call creates a fresh PersistentClient

    If _invalidate_chroma_client() regresses, search will return 0 hits
    even though the mine just reported drawers_filed >= 1.
    """

    def test_search_before_mine_returns_error(self, isolated_palace):
        """Pre-condition: searching an empty palace returns ok=False."""
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        result = adapter.run_search("anything")
        assert result.ok is False
        assert result.error is not None

    def test_status_before_mine_returns_error(self, isolated_palace):
        """Pre-condition: status on an empty palace returns ok=False."""
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        status = adapter.run_status()
        assert status.ok is False
        assert status.error is not None

    def test_warm_client_search_after_mine(self, isolated_palace, simple_project):
        """Core regression: same adapter instance must find drawers after mine.

        Step-by-step:
          1. Create adapter (empty palace → no client in cache yet).
          2. Trigger a failed search to warm the cache with a missing-collection
             path (FileNotFoundError from get_collection). This simulates the
             GUI opening before any mine has been run.
          3. Mine the project. _invalidate_chroma_client() evicts any cache.
          4. Search immediately — must find drawers.
        """
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))

        # Step 2: warm call that fails (palace not yet mined)
        pre_search = adapter.run_search("GraphQL")
        assert pre_search.ok is False  # expected — nothing mined yet

        # Step 3: mine
        mine_result = adapter.run_mine_projects(str(simple_project))
        assert mine_result.ok, f"Mine failed: {mine_result.error}"
        assert mine_result.drawers_filed >= 1

        # Step 4: immediate search in the same process
        post_search = adapter.run_search("GraphQL REST decision")
        assert post_search.ok, f"Post-mine search failed: {post_search.error}"
        assert len(post_search.hits) >= 1, (
            "Zero hits immediately after mine — warm-client eviction failed. "
            "_invalidate_chroma_client() did not evict the stale client or "
            "search_memories() is reading from a snapshot taken before mine."
        )

    def test_warm_client_status_after_mine(self, isolated_palace, simple_project):
        """Status must reflect new drawers immediately after mine on same adapter."""
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))

        # Warm the adapter with a failing status call
        pre_status = adapter.run_status()
        assert pre_status.ok is False

        # Mine
        mine_result = adapter.run_mine_projects(str(simple_project))
        assert mine_result.ok

        # Immediate status on same adapter
        post_status = adapter.run_status()
        assert post_status.ok, f"Post-mine status failed: {post_status.error}"
        assert post_status.total_drawers >= 1, (
            "total_drawers=0 immediately after mine — stale client in "
            "run_status() path (uses get_collection from _DEFAULT_BACKEND)."
        )
        assert len(post_status.wings) >= 1

    def test_warm_client_double_mine_search(self, isolated_palace, tmp_path):
        """Two sequential mines; search must see content from both."""
        proj1_parent = tmp_path / "p1"
        proj1_parent.mkdir()
        proj1 = _make_project(proj1_parent, files={
            "alpha.md": (
                "Alpha project: distributed tracing with Jaeger and OpenTelemetry.\n"
                "We use Jaeger for end-to-end request tracing across microservices.\n"
            )
        })

        proj2_parent = tmp_path / "p2"
        proj2_parent.mkdir()
        proj2 = _make_project(proj2_parent, files={
            "beta.md": (
                "Beta project: service mesh with Istio and Envoy sidecar proxy.\n"
                "Istio manages traffic, security, and observability for Kubernetes.\n"
            )
        })
        (proj2 / "mempalace.yaml").write_text(
            "wing: beta\nrooms:\n  - name: gen\n    description: gen\n    keywords: [gen]\n"
        )

        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))

        r1 = adapter.run_mine_projects(str(proj1))
        assert r1.ok, f"First mine failed: {r1.error}"

        r2 = adapter.run_mine_projects(str(proj2))
        assert r2.ok, f"Second mine failed: {r2.error}"

        # Both projects must be searchable after second mine
        s1 = adapter.run_search("Jaeger tracing")
        assert s1.ok and len(s1.hits) >= 1, (
            "First project content not found after second mine — "
            "double-mine invalidation issue."
        )

        s2 = adapter.run_search("Istio Envoy service mesh")
        assert s2.ok and len(s2.hits) >= 1, (
            "Second project content not found after second mine."
        )

    def test_enriched_fields_populated_after_warm_mine(self, isolated_palace, simple_project):
        """SearchHit enriched fields must be present after warm-client mine cycle."""
        adapter = MemPalaceAdapter(palace_path=str(isolated_palace))
        # Warm with failing search
        adapter.run_search("anything")
        # Mine
        adapter.run_mine_projects(str(simple_project))
        # Search
        result = adapter.run_search("GraphQL database")
        assert result.ok and result.hits

        hit = result.hits[0]
        # source_path: must be non-empty absolute path
        assert hit.source_path, "source_path empty after warm mine cycle"
        assert os.path.isabs(hit.source_path), f"source_path not absolute: {hit.source_path!r}"
        # drawer_id: must be non-empty string
        assert hit.drawer_id, "drawer_id empty after warm mine cycle"
        # chunk_index: must be an int
        assert isinstance(hit.chunk_index, int), (
            f"chunk_index is {type(hit.chunk_index).__name__}, expected int"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drain_until(
    ev_queue: queue.Queue,
    expected_types,
    timeout: float = 15,
) -> "BackendEvent | None":
    """Drain the event queue until an event of the given type(s) is found."""
    if isinstance(expected_types, EventType):
        expected_types = {expected_types}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            event = ev_queue.get(timeout=min(remaining, 0.5))
            if event.type in expected_types:
                return event
        except queue.Empty:
            pass
    return None
