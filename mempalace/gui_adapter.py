"""
gui_adapter.py — Backend adapter for macOS GUI (and any non-CLI caller).
=========================================================================

Wraps MemPalace Python functions into a clean, GUI-friendly API:

  - All operations return structured dataclasses, never raw prints.
  - Long-running operations (mine) stream progress via callbacks over a
    subprocess; mine is fully cancellable via MineHandle.cancel().
  - Interactive operations (init) are made non-interactive via --yes equivalents.
  - MCP server is managed as a subprocess with start/stop lifecycle.
  - All errors are returned as result objects with ok=False, never raised
    (except where explicitly documented as raising AdapterError).
  - Process-global environment is NOT mutated: palace_path is forwarded only
    via explicit CLI args and per-call env copies.

Usage example (future Swift/Flutter bridge):
    from mempalace.gui_adapter import MemPalaceAdapter
    adapter = MemPalaceAdapter(palace_path="~/Documents/my_palace")
    adapter.safe_init()
    handle = adapter.run_mine_projects("/Users/me/myproject", on_progress=print)
    # cancel from another thread: handle.cancel()
    results = adapter.run_search("why did we switch to GraphQL")
    status  = adapter.run_status()
    adapter.start_mcp_server()
    # ... later ...
    adapter.stop_mcp_server()

Design notes
------------
Stale ChromaDB client
    After run_mine_* completes, the subprocess has written new data to
    chroma.sqlite3.  The parent process caches a PersistentClient in
    palace.py's module-level _DEFAULT_BACKEND._clients dict; that cached
    client is now stale.  _invalidate_chroma_client() evicts the entry for
    this palace_path so the next search_memories() call opens a fresh client
    that sees the new drawers.

Mine subprocess timeout
    The naive approach of proc.wait(timeout=N) blocks on stdout reads first,
    so a hung child that stops printing but never exits would wait forever.
    Instead, a background reader thread drains stdout into a queue while the
    main thread enforces a wall-clock deadline on the whole operation.  When
    the deadline fires the process is killed, the reader thread is joined, and
    a MineResult(ok=False) is returned.

Mine output parsing
    Wing names and room names can contain spaces, hyphens, and punctuation.
    Regexes use .+ (greedy, whole-line match) rather than \\S+ for Wing.
    Room name lines are anchored to 4-space indent followed by 2+ spaces
    before the count, tolerating arbitrary room name characters.
    The JSONL sidecar env var MEMPALACE_GUI_PROGRESS is set to '1' in the
    child environment to signal future structured output (reserved for forward
    compatibility; current mine output is still human-readable text).

Environment isolation
    os.environ is never mutated.  Each subprocess call builds its own env
    dict from os.environ.copy() and passes it explicitly.  palace_path is
    forwarded via both the --palace CLI arg AND the env copy so child
    processes that import MempalaceConfig before parsing args still find it.
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("mempalace.gui_adapter")


# ---------------------------------------------------------------------------
# Interpreter resolution — ensures subprocess uses same env as parent
# ---------------------------------------------------------------------------


def _resolve_python() -> str:
    """Return the path to the Python interpreter that should be used for
    subprocess calls (mine, MCP server).

    The problem this solves: ``sys.executable`` can resolve to a bare
    ``/usr/bin/python3`` or the Xcode CLI tools Python
    (``/Library/Developer/CommandLineTools/usr/bin/python3``) when the
    GUI is launched via a ``.app`` bundle, a wrapper script, or
    ``python -m gui.app`` under a venv whose ``python`` is a symlink.

    Resolution strategy (first hit wins):
      0. **.app bundle detection** — if ``sys.frozen`` is set and
         ``sys.executable`` is inside an ``.app/Contents/MacOS/`` path,
         locate the real Python interpreter inside the bundled
         ``Python*.framework`` (py2app standalone mode).  The launcher
         at ``Contents/MacOS/AppName`` is NOT a general-purpose Python
         interpreter and cannot run ``-m`` flags.
      1. ``sys.executable`` — if it is a real, executable file and is
         *not* the Xcode CLI tools stub.  This covers normal venv usage.
      2. ``sys._base_executable`` — CPython sets this to the venv's
         underlying interpreter; it exists on macOS even when
         ``sys.executable`` points to a shim.
      3. ``shutil.which('python3')`` — fallback to PATH.

    Returns an absolute path string.  Logs a warning if the result seems
    dubious (e.g. Xcode CLI tools path).
    """
    import shutil

    XCODE_CLI_TOOLS = "/Library/Developer/CommandLineTools"

    def _is_good(path: str) -> bool:
        if not path:
            return False
        p = Path(path)
        if not p.is_file() or not os.access(path, os.X_OK):
            return False
        if XCODE_CLI_TOOLS in path:
            logger.warning(
                "Resolved Python is the Xcode CLI-tools stub (%s) — "
                "likely missing site-packages. Attempting fallback.",
                path,
            )
            return False
        return True

    # Step 0: py2app .app bundle — find the real Python in Frameworks/.
    if getattr(sys, "frozen", False) and ".app/Contents/MacOS/" in sys.executable:
        contents_dir = Path(sys.executable).resolve().parent.parent
        frameworks_dir = contents_dir / "Frameworks"
        if frameworks_dir.is_dir():
            for fw in sorted(frameworks_dir.glob("Python*.framework")):
                versions_dir = fw / "Versions"
                if not versions_dir.is_dir():
                    continue
                for ver_dir in sorted(versions_dir.iterdir(), reverse=True):
                    if not ver_dir.is_dir() or ver_dir.name == "Current":
                        continue
                    for py_name in (f"python3.{ver_dir.name}", "python3"):
                        py_bin = ver_dir / "bin" / py_name
                        if py_bin.is_file() and os.access(str(py_bin), os.X_OK):
                            logger.debug(
                                "Bundle Python interpreter: %s", py_bin
                            )
                            return str(py_bin)
        macos_python = contents_dir / "MacOS" / "python"
        if macos_python.is_file() and os.access(str(macos_python), os.X_OK):
            logger.debug("Bundle Python interpreter (MacOS/python): %s", macos_python)
            return str(macos_python)
        logger.warning(
            "Running inside .app bundle but could not find Python "
            "interpreter in Frameworks/ — subprocesses may fail."
        )

    candidates = [sys.executable]
    base = getattr(sys, "_base_executable", None)
    if base:
        candidates.append(base)
    candidates.append(shutil.which("python3") or "")

    for candidate in candidates:
        if _is_good(candidate):
            logger.debug("Resolved Python interpreter: %s", candidate)
            return candidate

    logger.warning(
        "Could not resolve a suitable Python interpreter; "
        "falling back to sys.executable=%s",
        sys.executable,
    )
    return sys.executable


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AdapterError(Exception):
    """Raised for unrecoverable configuration errors (wrong palace_path etc.)."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    """One result from a semantic search.

    Fields added in hardening pass 2.1:
      source_path   Full absolute path to the source file (stable across renames
                    of the palace dir).  source_file retains the basename for
                    display; source_path is for "open in editor" actions.
      drawer_id     ChromaDB document ID — stable reference for get_drawer calls.
      chunk_index   Zero-based chunk index within source_file (for pagination UI).
      closet_preview  First 200 chars of the closet entry that boosted this hit,
                       or None if this was a pure drawer (non-closet) match.
      line_start / line_end  1-based line range within the source file where
                    the chunk text occurs.  None when the file is not readable
                    or the text cannot be located (binary file, file deleted, etc.).
    """
    text: str
    wing: str
    room: str
    source_file: str
    source_path: str
    similarity: float
    distance: float
    matched_via: str = "drawer"
    drawer_id: str = ""
    chunk_index: Optional[int] = None
    closet_preview: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict, drawer_id: str = "") -> "SearchHit":
        return cls(
            text=d.get("text", ""),
            wing=d.get("wing", ""),
            room=d.get("room", ""),
            source_file=d.get("source_file", ""),
            source_path=d.get("source_path", ""),
            similarity=float(d.get("similarity", 0.0)),
            distance=float(d.get("distance", 1.0)),
            matched_via=d.get("matched_via", "drawer"),
            drawer_id=drawer_id or d.get("drawer_id", ""),
            chunk_index=d.get("chunk_index"),
            closet_preview=d.get("closet_preview"),
            line_start=d.get("line_start"),
            line_end=d.get("line_end"),
        )


@dataclass
class SearchResult:
    ok: bool
    query: str = ""
    hits: List[SearchHit] = field(default_factory=list)
    total_candidates: int = 0
    error: Optional[str] = None


@dataclass
class RoomCount:
    name: str
    drawers: int


@dataclass
class WingStatus:
    name: str
    rooms: List[RoomCount] = field(default_factory=list)

    @property
    def total_drawers(self) -> int:
        return sum(r.drawers for r in self.rooms)


@dataclass
class PalaceStatus:
    ok: bool
    total_drawers: int = 0
    total_files: int = 0
    wings: List[WingStatus] = field(default_factory=list)
    palace_path: str = ""
    chromadb_version: str = ""
    error: Optional[str] = None


@dataclass
class MineResult:
    ok: bool
    files_processed: int = 0
    files_skipped: int = 0
    drawers_filed: int = 0
    wing: str = ""
    rooms: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class InitResult:
    ok: bool
    config_path: str = ""
    palace_path: str = ""
    error: Optional[str] = None


@dataclass
class McpServerStatus:
    running: bool
    pid: Optional[int] = None
    palace_path: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# MineHandle — cancellable mine operation
# ---------------------------------------------------------------------------


class MineHandle:
    """Handle returned by run_mine_* for out-of-band cancellation.

    The caller can cancel from a different thread at any time::

        handle = adapter.run_mine_projects(dir, on_progress=cb)
        # from UI thread:
        handle.cancel()

    After cancel() the mine subprocess is killed (SIGKILL) and the
    run_mine_* call returns MineResult(ok=False, error='cancelled').
    """

    def __init__(self):
        self._cancel_event = threading.Event()

    def cancel(self):
        """Signal the mine to abort. Idempotent."""
        self._cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _wait(self, timeout: float) -> bool:
        """Return True if cancelled within timeout, False on timeout."""
        return self._cancel_event.wait(timeout)


# ---------------------------------------------------------------------------
# Progress line parser — parses mine stdout into structured updates
# ---------------------------------------------------------------------------

# "  Wing:    my cool project"  — Wing can contain spaces and special chars
_MINE_WING_RE = re.compile(r"Wing:\s+(.+)")
# "  ✓ [   1/42] filename with spaces.py                    +3"
_MINE_FILE_RE = re.compile(r"✓\s+\[\s*(\d+)/(\d+)\]\s+(.+?)\s+\+(\d+)\s*$")
# "  Files processed: 12"
_MINE_PROCESSED_RE = re.compile(r"Files processed:\s+(\d+)")
_MINE_SKIPPED_RE = re.compile(r"Files skipped.*?:\s+(\d+)")
_MINE_DRAWERS_RE = re.compile(r"Drawers filed:\s+(\d+)")
# "    arch & design        2 files"  — 4-space indent, room name, 2+ spaces, count
# Room names can contain spaces, hyphens, & ampersands, etc.
_MINE_ROOM_RE = re.compile(r"^    (.+?)\s{2,}(\d+)\s+files")


@dataclass
class MineProgressEvent:
    """Emitted per file during mine, and once at end."""
    type: str  # "file" | "done" | "error" | "info"
    # For type="file":
    current: int = 0
    total: int = 0
    filename: str = ""
    drawers: int = 0
    # For type="done":
    files_processed: int = 0
    files_skipped: int = 0
    drawers_filed: int = 0
    wing: str = ""
    rooms: Dict[str, int] = field(default_factory=dict)
    # For type="info" | "error":
    message: str = ""


def _parse_mine_line(line: str, state: dict) -> Optional[MineProgressEvent]:
    """Parse one stdout line from mine() into a MineProgressEvent, or None.

    Mine stdout format (relevant lines):
      '  Wing:    my cool project'           <- wing (may have spaces)
      '  ✓ [   1/42] file name.py    +3'    <- file progress
      '  Done.'                              <- summary begins
      '  Files processed: 12'
      '  Files skipped (already filed): 0'
      '  Drawers filed: 10'
      '    arch & design        N files'     <- room (may have special chars)
      '  Next: mempalace search ...'

    The 'done' event is emitted on the 'Next:' line — by then all summary
    counts (Files processed / Drawers filed / By room) have been parsed.

    Wing and room regexes use greedy .+ to handle names with spaces,
    hyphens, ampersands, and other punctuation.
    """
    m = _MINE_FILE_RE.search(line)
    if m:
        return MineProgressEvent(
            type="file",
            current=int(m.group(1)),
            total=int(m.group(2)),
            filename=m.group(3).strip(),
            drawers=int(m.group(4)),
        )

    m = _MINE_WING_RE.search(line)
    if m:
        state["wing"] = m.group(1).strip()

    m = _MINE_PROCESSED_RE.search(line)
    if m:
        state["files_processed"] = int(m.group(1))

    m = _MINE_SKIPPED_RE.search(line)
    if m:
        state["files_skipped"] = int(m.group(1))

    m = _MINE_DRAWERS_RE.search(line)
    if m:
        state["drawers_filed"] = int(m.group(1))

    m = _MINE_ROOM_RE.match(line)
    if m:
        room_name = m.group(1).strip()
        state.setdefault("rooms", {})[room_name] = int(m.group(2))

    if "Done." in line:
        state["seen_done"] = True

    # Emit done event on the 'Next:' line — all counts parsed by now.
    if "Next:" in line and state.get("seen_done"):
        return MineProgressEvent(
            type="done",
            files_processed=state.get("files_processed", 0),
            files_skipped=state.get("files_skipped", 0),
            drawers_filed=state.get("drawers_filed", 0),
            wing=state.get("wing", ""),
            rooms=dict(state.get("rooms", {})),
        )

    return None


# ---------------------------------------------------------------------------
# Stale-client eviction
# ---------------------------------------------------------------------------


def _invalidate_chroma_client(palace_path: str) -> None:
    """Evict the cached ChromaDB PersistentClient for *palace_path*.

    palace.py holds a module-level _DEFAULT_BACKEND = ChromaBackend() whose
    _clients dict caches one PersistentClient per palace path.  After a
    subprocess mine writes new drawers, that cached client is stale — it
    holds an in-memory HNSW index snapshot from before the mine.  Evicting
    the entry forces the next get_collection() call to open a fresh client
    that reads the updated SQLite file.

    This mirrors the approach in mcp_server.py (_get_client uses inode/mtime
    checks) but uses explicit eviction because the adapter controls when writes
    happen (after run_mine_* returns).

    Safe to call even if the palace was never opened in this process
    (evicting a non-existent key is a no-op).
    """
    try:
        from .palace import _DEFAULT_BACKEND
        _DEFAULT_BACKEND._clients.pop(palace_path, None)
        logger.debug("Evicted stale ChromaDB client for %s", palace_path)
    except Exception as e:
        logger.warning("Could not evict Chroma client cache: %s", e)


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------


class MemPalaceAdapter:
    """
    GUI-safe backend adapter for MemPalace.

    All public methods return structured result objects or dicts.
    No method prints to stdout or mutates os.environ.

    Thread safety
    -------------
    run_search / run_status: direct Python calls, safe to call serially
        from a single thread.  ChromaDB 1.5.x uses process-level file locking;
        do NOT call these concurrently from multiple threads pointing at the
        same palace (use a single serial worker thread or asyncio task).
    run_mine_*: subprocess — fully isolated from the parent process's Chroma
        client.  Returns a MineHandle for out-of-band cancellation.
    start/stop_mcp_server: thread-safe via _mcp_lock.
    """

    def __init__(self, palace_path: Optional[str] = None):
        """
        Args:
            palace_path: Override palace location. If None, uses MempalaceConfig
                         (env MEMPALACE_PALACE_PATH > ~/.mempalace/config.json >
                          ~/.mempalace/palace).

        Note: palace_path is stored internally and passed via CLI args and
        per-call env copies to subprocesses.  os.environ is NOT mutated.
        """
        from .config import MempalaceConfig

        self._config = MempalaceConfig()
        if palace_path:
            self._palace_path = str(Path(palace_path).expanduser().resolve())
        else:
            self._palace_path = self._config.palace_path

        self._mcp_proc: Optional[subprocess.Popen] = None
        self._mcp_lock = threading.Lock()

    @property
    def palace_path(self) -> str:
        return self._palace_path

    def switch_palace(self, new_path: str) -> str:
        """Switch the adapter to a different palace path.

        Stops the MCP server if running, invalidates the cached ChromaDB
        client for the old path, and updates the internal palace_path.

        Returns the new palace path.
        """
        if new_path == self._palace_path:
            return self._palace_path
        self.stop_mcp_server()
        _invalidate_chroma_client(self._palace_path)
        self._palace_path = str(Path(new_path).expanduser().resolve())
        logger.info("Switched palace: %s", self._palace_path)
        return self._palace_path

    # ------------------------------------------------------------------
    # _child_env — build isolated subprocess environment
    # ------------------------------------------------------------------

    def _child_env(self, extra: Optional[dict] = None) -> dict:
        """Return a copy of os.environ augmented with MemPalace settings.

        Does NOT mutate os.environ.  palace_path is forwarded via
        MEMPALACE_PALACE_PATH so child processes that read config before
        parsing --palace see the right value.

        PYTHONPATH is extended with the parent's ``sys.path`` so that a
        child subprocess using a different (or system) Python interpreter
        can still find packages installed in the parent's environment
        (e.g. ``pip install --user`` chromadb into
        ``~/Library/Python/3.X/lib/python/site-packages``).

        Inside a .app bundle:

        * ``PYTHONHOME`` is set to the bundle's Resources directory so
          the ``python`` binary in ``Contents/MacOS/`` can find its
          standard library and site-packages.  Without this, the
          subprocess python falls back to the system Python's prefix.
        * The inherited ``PYTHONPATH`` from the parent environment is
          replaced (not extended) with only the bundle's ``sys.path``
          entries.  This prevents stale development-environment paths
          from leaking into the subprocess and shadowing the bundled
          packages.
        """
        env = os.environ.copy()
        env["MEMPALACE_PALACE_PATH"] = self._palace_path
        env["PYTHONUNBUFFERED"] = "1"
        env["MEMPALACE_GUI_PROGRESS"] = "1"
        parent_paths = [p for p in sys.path if p]
        if not parent_paths:
            if extra:
                env.update(extra)
            return env
        in_bundle = getattr(sys, "frozen", False) and ".app/Contents/MacOS/" in sys.executable
        if in_bundle:
            resources_dir = str(Path(sys.executable).resolve().parent.parent / "Resources")
            env["PYTHONHOME"] = resources_dir
            env["PYTHONPATH"] = os.pathsep.join(parent_paths)
        else:
            combined = os.pathsep.join(parent_paths)
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                combined = existing_pythonpath + os.pathsep + combined
            env["PYTHONPATH"] = combined
        if extra:
            env.update(extra)
        return env

    # ------------------------------------------------------------------
    # safe_init — non-interactive palace initialization
    # ------------------------------------------------------------------

    def safe_init(
        self,
        project_dir: Optional[str] = None,
        *,
        auto_detect_rooms: bool = False,
    ) -> InitResult:
        """Initialize MemPalace without any interactive prompts.

        Steps:
          1. Create ~/.mempalace/ config dir and config.json (always safe).
          2. Optionally run room detection on project_dir with yes=True
             (all input() calls suppressed).

        Args:
            project_dir: If provided, detect rooms from this directory.
                         mempalace.yaml will be written there.
            auto_detect_rooms: If True and project_dir given, run room
                               detection with auto-accept (--yes mode).

        Returns:
            InitResult with ok=True on success.
        """
        try:
            config_path = self._config.init()
        except OSError as e:
            return InitResult(ok=False, error=f"Could not create config dir: {e}")

        if project_dir and auto_detect_rooms:
            try:
                from .room_detector_local import detect_rooms_local

                detect_rooms_local(project_dir=project_dir, yes=True)
            except Exception as e:
                logger.warning("Room detection failed (non-fatal): %s", e)
                # Non-fatal: palace can still be used without mempalace.yaml

        return InitResult(
            ok=True,
            config_path=str(config_path),
            palace_path=self._palace_path,
        )

    # ------------------------------------------------------------------
    # run_search — direct Python API, returns structured data
    # ------------------------------------------------------------------

    def run_search(
        self,
        query: str,
        *,
        wing: Optional[str] = None,
        room: Optional[str] = None,
        n_results: int = 5,
        max_distance: float = 1.0,
    ) -> SearchResult:
        """Semantic search against the palace. Returns structured SearchResult.

        This is a direct Python call — no subprocess, no stdout.
        Call from a single dedicated thread; do not issue concurrent searches
        against the same palace (ChromaDB 1.5.x is not concurrent-client safe).

        The palace's Chroma client is always fresh after run_mine_* because
        _invalidate_chroma_client() is called at the end of every mine.

        Args:
            query: Natural language query.
            wing: Optional wing filter.
            room: Optional room filter.
            n_results: Max results to return.
            max_distance: Cosine distance cutoff (0=identical, 2=opposite).
                          0.0 disables filtering. Practical thresholds:
                            0.8 — strict (high relevance)
                            1.0 — balanced (default, similarity > 0)
                            1.5 — permissive (weak matches included)

        Returns:
            SearchResult with ok=True and hits list on success,
            or ok=False with error string if palace not found.
        """
        try:
            from .searcher import search_memories
        except ImportError as e:
            return SearchResult(ok=False, query=query, error=f"Import error: {e}")

        try:
            raw = search_memories(
                query=query,
                palace_path=self._palace_path,
                wing=wing,
                room=room,
                n_results=n_results,
                max_distance=max_distance,
            )
        except Exception as e:
            return SearchResult(ok=False, query=query, error=str(e))

        if "error" in raw:
            return SearchResult(ok=False, query=query, error=raw["error"])

        # Enrich hits with full source path and drawer_id from a second lookup.
        # search_memories() strips _source_file_full and _chunk_index from its
        # output; we re-fetch them via a direct metadata get keyed by source_file.
        enriched = self._enrich_hits(raw.get("results", []))

        hits = [SearchHit.from_dict(h) for h in enriched]
        self._compute_line_ranges(hits)
        return SearchResult(
            ok=True,
            query=raw.get("query", query),
            hits=hits,
            total_candidates=raw.get("total_before_filter", len(hits)),
        )

    def _enrich_hits(self, results: list) -> list:
        """Ensure source_path, drawer_id, chunk_index are present in every hit.

        Since searcher.search_memories() now promotes the internal
        ``_source_file_full``, ``_chunk_index``, and the ChromaDB ``ids``
        to ``source_path``, ``chunk_index``, and ``drawer_id`` directly, this
        method is now a simple normalisation pass — it fills in empty defaults
        for any field that is missing (e.g. when called against an older
        searcher that hasn't been upgraded yet).

        The former basename-suffix reverse-lookup via collection.get() has been
        removed because:
          * It was O(N) round-trips per hit.
          * Basename matching silently collides when two files in different
            directories share a filename.
          * The data is now available without a second round-trip.

        On any failure this degrades gracefully: missing fields remain "".
        """
        enriched = []
        for h in results:
            h = dict(h)  # shallow copy — don't mutate the caller's dict
            h.setdefault("source_path", "")
            h.setdefault("drawer_id", "")
            h.setdefault("chunk_index", None)
            enriched.append(h)
        return enriched

    @staticmethod
    def _compute_line_ranges(hits: list) -> None:
        """Populate line_start / line_end on hits where the source file exists.

        For each hit whose source_path points to a readable text file, the
        method attempts to locate the hit's chunk text within the file and
        compute a 1-based line range.  If the text cannot be located (binary
        file, file deleted, modified since mining, etc.) the fields remain
        None — the caller can fall back to chunk_index for display.

        The match strategy is:
          1. Take the first distinctive line of the chunk (>= 20 chars, not
             blank) as an anchor.
          2. Scan the file for that anchor line.
          3. From the anchor position, compute the line range by counting
             the number of lines in the chunk text.

        This is intentionally simple — IDE-level navigation is out of scope.
        """
        for hit in hits:
            if not hit.source_path or hit.line_start is not None:
                continue
            try:
                p = Path(hit.source_path)
                if not p.is_file():
                    continue
                content = p.read_text(errors="replace")
                if not content:
                    continue
                lines = content.splitlines()
                chunk_lines = hit.text.splitlines()

                anchor = None
                anchor_offset = 0
                for i, cl in enumerate(chunk_lines):
                    stripped = cl.strip()
                    if len(stripped) >= 20:
                        anchor = stripped
                        anchor_offset = i
                        break
                if not anchor:
                    continue

                for fi, fl in enumerate(lines):
                    if anchor in fl:
                        start = fi - anchor_offset + 1
                        start = max(1, start)
                        end = start + len(chunk_lines) - 1
                        end = min(end, len(lines))
                        hit.line_start = start
                        hit.line_end = end
                        break
            except Exception:
                pass

    # ------------------------------------------------------------------
    # run_status — captures stdout, returns structured PalaceStatus
    # ------------------------------------------------------------------

    def run_status(self) -> PalaceStatus:
        """Return palace status as a structured object (no stdout printed).

        Reads ChromaDB metadata directly. Fast, non-blocking.

        Returns:
            PalaceStatus with wings/rooms breakdown, or ok=False with error.
        """
        try:
            import chromadb as _chromadb
            chromadb_version = _chromadb.__version__
        except ImportError:
            chromadb_version = "unknown"

        try:
            from .palace import get_collection
            col = get_collection(self._palace_path, create=False)
        except FileNotFoundError:
            return PalaceStatus(
                ok=False,
                palace_path=self._palace_path,
                chromadb_version=chromadb_version,
                error="No palace found. Run safe_init() then run_mine_projects().",
            )
        except Exception as e:
            return PalaceStatus(
                ok=False,
                palace_path=self._palace_path,
                chromadb_version=chromadb_version,
                error=str(e),
            )

        try:
            total = col.count()
            if total == 0:
                return PalaceStatus(
                    ok=True,
                    total_drawers=0,
                    palace_path=self._palace_path,
                    chromadb_version=chromadb_version,
                )

            # Paginate to avoid 10K silent truncation
            all_meta = []
            batch_size = 1000
            offset = 0
            while offset < total:
                batch = col.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"],
                )
                if not batch.get("metadatas"):
                    break
                all_meta.extend(batch["metadatas"])
                offset += len(batch["metadatas"])

            # Aggregate wing -> room -> count
            wing_room: Dict[str, Dict[str, int]] = {}
            source_files: set = set()
            for m in all_meta:
                w = m.get("wing", "unknown")
                r = m.get("room", "unknown")
                wing_room.setdefault(w, {}).setdefault(r, 0)
                wing_room[w][r] += 1
                sf = m.get("source_file", "")
                if sf:
                    source_files.add(sf)

            wings = [
                WingStatus(
                    name=w,
                    rooms=[RoomCount(name=r, drawers=c) for r, c in sorted(rooms.items())],
                )
                for w, rooms in sorted(wing_room.items())
            ]

            return PalaceStatus(
                ok=True,
                total_drawers=total,
                total_files=len(source_files),
                wings=wings,
                palace_path=self._palace_path,
                chromadb_version=chromadb_version,
            )
        except Exception as e:
            return PalaceStatus(
                ok=False,
                palace_path=self._palace_path,
                chromadb_version=chromadb_version,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # run_mine_projects / run_mine_convos — subprocess with cancellable timeout
    # ------------------------------------------------------------------

    def run_mine_projects(
        self,
        project_dir: str,
        *,
        wing: Optional[str] = None,
        agent: str = "mempalace",
        limit: int = 0,
        dry_run: bool = False,
        respect_gitignore: bool = True,
        on_progress: Optional[Callable[[MineProgressEvent], None]] = None,
        timeout: Optional[float] = None,
        handle: Optional[MineHandle] = None,
    ) -> MineResult:
        """Mine a project directory into the palace via subprocess.

        Runs in a subprocess so it can be safely cancelled without blocking
        the GUI event loop. Progress events are streamed via the on_progress
        callback as each file is processed.

        A background thread drains subprocess stdout; the calling thread
        enforces a wall-clock deadline so a hung child (stops printing but
        never exits) is killed within `timeout` seconds, not blocked forever.

        After the mine completes the in-process ChromaDB client cache is
        invalidated so subsequent run_search() calls see the new drawers.

        Args:
            project_dir: Directory to mine.
            wing: Override wing name (default: directory basename).
            agent: Agent name recorded on drawers.
            limit: Max files to process (0 = all).
            dry_run: Preview without filing.
            respect_gitignore: Honour .gitignore rules.
            on_progress: Callback(MineProgressEvent) called per file and on done.
            timeout: Max wall-clock seconds for the entire mine (None = no limit).
            handle: Optional MineHandle for out-of-band cancellation from another
                    thread (e.g. a Cancel button in the GUI).

        Returns:
            MineResult with counts, or ok=False with error.
        """
        return self._run_mine(
            mode="projects",
            source_dir=project_dir,
            wing=wing,
            agent=agent,
            limit=limit,
            dry_run=dry_run,
            respect_gitignore=respect_gitignore,
            on_progress=on_progress,
            timeout=timeout,
            handle=handle,
        )

    def run_mine_convos(
        self,
        convo_dir: str,
        *,
        wing: Optional[str] = None,
        agent: str = "mempalace",
        limit: int = 0,
        dry_run: bool = False,
        extract_mode: str = "exchange",
        on_progress: Optional[Callable[[MineProgressEvent], None]] = None,
        timeout: Optional[float] = None,
        handle: Optional[MineHandle] = None,
    ) -> MineResult:
        """Mine conversation exports (Claude, ChatGPT, Slack) via subprocess.

        Args:
            convo_dir: Directory with chat export files.
            wing: Override wing name.
            agent: Agent name.
            limit: Max files (0 = all).
            dry_run: Preview without filing.
            extract_mode: "exchange" (default) or "general".
            on_progress: Callback(MineProgressEvent).
            timeout: Max wall-clock seconds (None = no limit).
            handle: Optional MineHandle for out-of-band cancellation.

        Returns:
            MineResult with counts, or ok=False with error.
        """
        return self._run_mine(
            mode="convos",
            source_dir=convo_dir,
            wing=wing,
            agent=agent,
            limit=limit,
            dry_run=dry_run,
            extract_mode=extract_mode,
            on_progress=on_progress,
            timeout=timeout,
            handle=handle,
        )

    def _run_mine(
        self,
        mode: str,
        source_dir: str,
        wing: Optional[str],
        agent: str,
        limit: int,
        dry_run: bool,
        respect_gitignore: bool = True,
        extract_mode: str = "exchange",
        on_progress: Optional[Callable[[MineProgressEvent], None]] = None,
        timeout: Optional[float] = None,
        handle: Optional[MineHandle] = None,
    ) -> MineResult:
        """Internal: build CLI command, drain stdout in a background thread,
        enforce wall-clock deadline, invalidate stale Chroma client."""
        cmd = [
            _resolve_python(), "-m", "mempalace",
            "--palace", self._palace_path,
            "mine", source_dir,
            "--mode", mode,
            "--agent", agent,
        ]
        if wing:
            cmd += ["--wing", wing]
        if limit > 0:
            cmd += ["--limit", str(limit)]
        if dry_run:
            cmd.append("--dry-run")
        if not respect_gitignore:
            cmd.append("--no-gitignore")
        if mode == "convos":
            cmd += ["--extract", extract_mode]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=self._child_env(),
                bufsize=1,
            )
        except Exception as e:
            return MineResult(ok=False, error=f"Failed to start mine subprocess: {e}")

        # Background reader thread: drains stdout into a queue so the main
        # thread is never blocked waiting for a line that never comes.
        line_queue: queue.Queue = queue.Queue()

        def _reader():
            try:
                for line in proc.stdout:
                    line_queue.put(line.rstrip("\n"))
            except Exception:
                pass
            finally:
                line_queue.put(None)  # sentinel: EOF

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        state: dict = {}
        last_done_event: Optional[MineProgressEvent] = None
        all_lines: list = []
        deadline = (time.monotonic() + timeout) if timeout else None

        try:
            while True:
                # Check external cancellation
                if handle and handle.cancelled:
                    proc.kill()
                    proc.wait()
                    reader.join(timeout=2)
                    return MineResult(ok=False, error="Mine cancelled")

                # Compute remaining time
                remaining = None
                if deadline:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        proc.kill()
                        proc.wait()
                        reader.join(timeout=2)
                        return MineResult(
                            ok=False,
                            error=f"Mine timed out after {timeout:.0f}s",
                        )

                try:
                    # Poll queue with a short tick so cancellation is responsive
                    tick = min(0.1, remaining) if remaining else 0.1
                    line = line_queue.get(timeout=tick)
                except queue.Empty:
                    continue

                if line is None:
                    break  # EOF sentinel from reader thread

                all_lines.append(line)
                event = _parse_mine_line(line, state)
                if event:
                    if event.type == "done":
                        last_done_event = event
                    if on_progress:
                        try:
                            on_progress(event)
                        except Exception:
                            pass  # never let callback crash the adapter

        except Exception as e:
            proc.kill()
            proc.wait()
            reader.join(timeout=2)
            return MineResult(ok=False, error=f"Mine subprocess error: {e}")

        proc.wait()
        reader.join(timeout=5)

        # Always evict the stale Chroma client so the next search sees new data.
        if not dry_run:
            _invalidate_chroma_client(self._palace_path)

        if proc.returncode != 0:
            tail = "\n".join(all_lines[-10:])
            return MineResult(ok=False, error=f"Mine exited {proc.returncode}:\n{tail}")

        if last_done_event:
            return MineResult(
                ok=True,
                files_processed=last_done_event.files_processed,
                files_skipped=last_done_event.files_skipped,
                drawers_filed=last_done_event.drawers_filed,
                wing=last_done_event.wing,
                rooms=last_done_event.rooms,
            )

        # Fallback if "Done." line was not detected (dry_run or empty project)
        return MineResult(
            ok=True,
            files_processed=state.get("files_processed", 0),
            files_skipped=state.get("files_skipped", 0),
            drawers_filed=state.get("drawers_filed", 0),
            wing=state.get("wing", wing or ""),
            rooms=state.get("rooms", {}),
        )

    # ------------------------------------------------------------------
    # MCP server lifecycle
    # ------------------------------------------------------------------

    def start_mcp_server(
        self,
        *,
        extra_args: Optional[List[str]] = None,
        stderr_log: Optional[str] = None,
    ) -> McpServerStatus:
        """Start the MemPalace MCP server as a managed subprocess.

        The server reads JSON-RPC from stdin and writes to stdout.
        This adapter keeps a handle to the process for later stop/status checks.
        Only one server instance is managed at a time per adapter instance.

        Args:
            extra_args: Additional CLI args (e.g. ["--palace", "/custom/path"]).
            stderr_log: Path to write server stderr logs to. If None, stderr
                        is discarded. Pass "/dev/stderr" to see logs in terminal.

        Returns:
            McpServerStatus with running=True and pid on success.
        """
        with self._mcp_lock:
            if self._mcp_proc and self._mcp_proc.poll() is None:
                return McpServerStatus(
                    running=True,
                    pid=self._mcp_proc.pid,
                    palace_path=self._palace_path,
                )

            cmd = [
                _resolve_python(), "-m", "mempalace.mcp_server",
                "--palace", self._palace_path,
            ]
            if extra_args:
                cmd.extend(extra_args)

            if stderr_log:
                stderr_fh = open(stderr_log, "a", encoding="utf-8")
            else:
                stderr_fh = subprocess.DEVNULL

            try:
                self._mcp_proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_fh,
                    text=True,
                    env=self._child_env(),
                    bufsize=1,
                )
            except Exception as e:
                return McpServerStatus(
                    running=False,
                    palace_path=self._palace_path,
                    error=f"Failed to start MCP server: {e}",
                )

            # Brief readiness check — verify server responds to initialize
            try:
                import select

                init_req = json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                    },
                })
                self._mcp_proc.stdin.write(init_req + "\n")
                self._mcp_proc.stdin.flush()

                ready, _, _ = select.select([self._mcp_proc.stdout], [], [], 3.0)
                if ready:
                    line = self._mcp_proc.stdout.readline()
                    resp = json.loads(line)
                    if "result" not in resp:
                        raise ValueError(f"Unexpected response: {resp}")
                    logger.info("MCP server ready (pid=%d)", self._mcp_proc.pid)
                else:
                    logger.warning("MCP server did not respond to initialize within 3s")

            except Exception as e:
                logger.warning("MCP readiness check failed (server may still work): %s", e)

            return McpServerStatus(
                running=self._mcp_proc.poll() is None,
                pid=self._mcp_proc.pid,
                palace_path=self._palace_path,
            )

    def stop_mcp_server(self, *, timeout: float = 5.0) -> McpServerStatus:
        """Stop the managed MCP server subprocess.

        Sends SIGTERM first, then SIGKILL if it doesn't exit within timeout.

        Args:
            timeout: Seconds to wait for graceful shutdown before SIGKILL.

        Returns:
            McpServerStatus with running=False on success.
        """
        with self._mcp_lock:
            if not self._mcp_proc or self._mcp_proc.poll() is not None:
                return McpServerStatus(running=False, palace_path=self._palace_path)

            try:
                self._mcp_proc.terminate()
                try:
                    self._mcp_proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._mcp_proc.kill()
                    self._mcp_proc.wait()
                logger.info("MCP server stopped (pid was %d)", self._mcp_proc.pid)
                self._mcp_proc = None
            except Exception as e:
                return McpServerStatus(
                    running=True,
                    pid=self._mcp_proc.pid if self._mcp_proc else None,
                    palace_path=self._palace_path,
                    error=f"Stop failed: {e}",
                )

            return McpServerStatus(running=False, palace_path=self._palace_path)

    def mcp_server_status(self) -> McpServerStatus:
        """Return current MCP server status without starting or stopping it."""
        with self._mcp_lock:
            if not self._mcp_proc:
                return McpServerStatus(running=False, palace_path=self._palace_path)
            alive = self._mcp_proc.poll() is None
            return McpServerStatus(
                running=alive,
                pid=self._mcp_proc.pid if alive else None,
                palace_path=self._palace_path,
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def check_chromadb_version() -> dict:
        """Check ChromaDB version against minimum safe requirement.

        Returns:
            {
              "version": "1.5.7",
              "ok": True,
              "minimum": "1.5.4",
              "arm64_safe": True,     # True if >= 1.5.4 (Rust, no hnswlib)
              "warning": None | "...",
            }
        """
        MIN_SAFE = (1, 5, 4)
        try:
            import chromadb

            raw = chromadb.__version__
            parts = tuple(int(x) for x in raw.split(".")[:3])
            ok = parts >= MIN_SAFE
            return {
                "version": raw,
                "ok": ok,
                "minimum": "1.5.4",
                "arm64_safe": ok,
                "warning": (
                    None
                    if ok
                    else (
                        f"ChromaDB {raw} is below minimum 1.5.4. "
                        "Segfaults likely on macOS ARM64. "
                        "Run: pip install 'chromadb>=1.5.4'"
                    )
                ),
            }
        except ImportError:
            return {
                "version": None,
                "ok": False,
                "minimum": "1.5.4",
                "arm64_safe": False,
                "warning": "chromadb is not installed.",
            }

    @staticmethod
    def check_palace_exists(palace_path: str) -> dict:
        """Check if a palace directory contains a valid database.

        Returns:
            {"exists": bool, "has_db": bool, "path": str}
        """
        p = Path(palace_path).expanduser().resolve()
        db = p / "chroma.sqlite3"
        return {
            "exists": p.is_dir(),
            "has_db": db.is_file(),
            "path": str(p),
        }

    def __repr__(self) -> str:
        mcp_pid = self._mcp_proc.pid if self._mcp_proc and self._mcp_proc.poll() is None else None
        return (
            f"MemPalaceAdapter(palace_path={self._palace_path!r}, "
            f"mcp_pid={mcp_pid})"
        )
