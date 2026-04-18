"""
backend_controller.py — Single execution boundary for the MemPalace GUI backend.
=================================================================================

Design goals
------------
* **One worker thread** handles all blocking operations (mine, search, status,
  MCP server control) so the GUI event loop is never blocked.
* **Typed events** are posted back to the caller via a thread-safe callback
  (``on_event``) or, when desired, retrieved from the public ``events`` queue.
* **No PySide6 / Qt / UI imports** — this module only imports stdlib + the
  MemPalace adapter layer.  The GUI layer wraps it; this file stays portable.
* **Cancellable mine** — the caller receives a ``MineHandle`` and can call
  ``handle.cancel()`` from any thread; the worker honours it.
* **Serial search** — only one search runs at a time; concurrent ``search``
  calls are serialised through the same worker queue, so ChromaDB is never
  hit by two threads simultaneously.

Usage example (future Swift/Flutter bridge or PySide6 app)::

    from mempalace.backend_controller import BackendController, EventType

    def handle_event(event):
        if event.type == EventType.SEARCH_DONE:
            print(event.payload)
        elif event.type == EventType.MINE_PROGRESS:
            print(event.payload.filename)

    ctrl = BackendController(palace_path="~/my_palace", on_event=handle_event)
    ctrl.start()               # starts the worker thread

    ctrl.search("GraphQL")
    handle = ctrl.mine("/path/to/project")
    # From a Cancel button:
    handle.cancel()

    ctrl.stop()                # graceful shutdown

Event contract
--------------
Every public method posts a *start* event immediately (in the calling thread),
then the worker posts a *done* or *error* event when it finishes.  Mine also
posts per-file ``MINE_PROGRESS`` events from within the worker.

    SEARCH_STARTED  → payload = {"query": str}
    SEARCH_DONE     → payload = SearchResult
    SEARCH_ERROR    → payload = {"error": str}

    MINE_STARTED    → payload = {"source_dir": str, "mode": str}
    MINE_PROGRESS   → payload = MineProgressEvent
    MINE_DONE       → payload = MineResult
    MINE_ERROR      → payload = {"error": str}

    STATUS_STARTED  → payload = {}
    STATUS_DONE     → payload = PalaceStatus
    STATUS_ERROR    → payload = {"error": str}

    MCP_STARTED     → payload = McpServerStatus
    MCP_STOPPED     → payload = McpServerStatus

    CONTROLLER_ERROR → payload = {"error": str}   (unhandled exception in worker)
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from .gui_adapter import (
    MemPalaceAdapter,
    MineHandle,
    MineProgressEvent,
    MineResult,
    PalaceStatus,
    SearchResult,
    McpServerStatus,
)

logger = logging.getLogger("mempalace.backend_controller")


# ---------------------------------------------------------------------------
# Event types & container
# ---------------------------------------------------------------------------


class EventType(Enum):
    # Search lifecycle
    SEARCH_STARTED = auto()
    SEARCH_DONE = auto()
    SEARCH_ERROR = auto()

    # Mine lifecycle
    MINE_STARTED = auto()
    MINE_PROGRESS = auto()
    MINE_DONE = auto()
    MINE_ERROR = auto()

    # Status lifecycle
    STATUS_STARTED = auto()
    STATUS_DONE = auto()
    STATUS_ERROR = auto()

    # MCP server lifecycle
    MCP_STARTED = auto()
    MCP_STOPPED = auto()

    # Worker-level error (unhandled exception)
    CONTROLLER_ERROR = auto()


@dataclass
class BackendEvent:
    """A typed event posted from the BackendController worker to the caller."""
    type: EventType
    payload: Any = None
    # Optional request-id so the GUI can correlate a *_DONE event with the
    # original *_STARTED event that triggered it (e.g. for progress-bar updates
    # where multiple searches may be in-flight in the queue).
    request_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Work items posted into the worker queue
# ---------------------------------------------------------------------------


@dataclass
class _WorkItem:
    """Internal: describes one unit of work for the worker thread."""
    kind: str             # "search" | "mine" | "status" | "mcp_start" | "mcp_stop" | "shutdown"
    kwargs: dict = field(default_factory=dict)
    request_id: Optional[int] = None
    mine_handle: Optional[MineHandle] = None   # for "mine" kind only


# ---------------------------------------------------------------------------
# BackendController
# ---------------------------------------------------------------------------


class BackendController:
    """
    Single-threaded execution boundary for all MemPalace backend operations.

    One instance → one daemon worker thread → one ChromaDB client lifetime.
    The GUI creates *one* BackendController and routes all operations through
    it.  This guarantees:

    * No concurrent ChromaDB reads/writes from the same process.
    * Mines are safely cancellable via MineHandle.cancel().
    * Events are dispatched in chronological order to ``on_event``.

    Thread safety
    -------------
    * ``search()``, ``mine()``, ``status()``, ``start_mcp_server()``,
      ``stop_mcp_server()`` are all thread-safe — they just enqueue work.
    * ``on_event`` is called from the **worker thread**.  If you integrate with
      Qt / PySide6, emit a Qt signal inside ``on_event`` (``Signal.emit`` is
      thread-safe) rather than touching widgets directly.
    * ``events`` (the public ``queue.Queue``) is thread-safe for
      ``get()``/``get_nowait()`` from the caller thread.
    """

    def __init__(
        self,
        palace_path: Optional[str] = None,
        *,
        on_event: Optional[Callable[[BackendEvent], None]] = None,
        worker_queue_maxsize: int = 64,
    ):
        """
        Args:
            palace_path: Palace directory (None = use MempalaceConfig default).
            on_event: Callback invoked for every BackendEvent from the worker
                      thread.  If None, events are only posted to ``self.events``.
            worker_queue_maxsize: Max pending work items before ``search()`` etc.
                                  block.  64 is conservative; increase if the GUI
                                  submits bursts of work (e.g. live-search typing).
        """
        self._adapter = MemPalaceAdapter(palace_path=palace_path)
        self._on_event = on_event

        # Public event queue — callers may poll this if they prefer pull over push.
        self.events: queue.Queue[BackendEvent] = queue.Queue()

        self._work_q: queue.Queue[_WorkItem] = queue.Queue(maxsize=worker_queue_maxsize)
        self._worker_thread: Optional[threading.Thread] = None
        self._request_counter = 0
        self._counter_lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def palace_path(self) -> str:
        return self._adapter.palace_path

    def start(self) -> "BackendController":
        """Start the background worker thread. Idempotent."""
        if self._started:
            return self
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="mempalace-backend-worker",
            daemon=True,
        )
        self._worker_thread.start()
        self._started = True
        logger.info("BackendController worker started (palace=%s)", self.palace_path)
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully stop the worker thread.

        Sends a sentinel shutdown item and waits up to ``timeout`` seconds.
        Outstanding work items already in the queue may complete before shutdown
        is processed (FIFO ordering).
        """
        if not self._started:
            return
        self._work_q.put(_WorkItem(kind="shutdown"))
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        self._started = False
        logger.info("BackendController worker stopped")

    # ------------------------------------------------------------------
    # Public API — enqueue work, return immediately
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        wing: Optional[str] = None,
        room: Optional[str] = None,
        n_results: int = 5,
    ) -> int:
        """Enqueue a search. Returns request_id for correlation.

        Posts SEARCH_STARTED immediately, then SEARCH_DONE (or SEARCH_ERROR)
        from the worker when results are ready.
        """
        rid = self._next_rid()
        self._post(BackendEvent(type=EventType.SEARCH_STARTED, payload={"query": query}, request_id=rid))
        self._work_q.put(_WorkItem(
            kind="search",
            kwargs=dict(query=query, wing=wing, room=room, n_results=n_results),
            request_id=rid,
        ))
        return rid

    def mine(
        self,
        source_dir: str,
        *,
        mode: str = "projects",
        wing: Optional[str] = None,
        agent: str = "mempalace",
        limit: int = 0,
        dry_run: bool = False,
        respect_gitignore: bool = True,
        extract_mode: str = "exchange",
        timeout: Optional[float] = None,
    ) -> tuple[MineHandle, int]:
        """Enqueue a mine operation. Returns (MineHandle, request_id).

        The caller can call ``handle.cancel()`` from any thread to abort the
        mine.  Posts MINE_STARTED immediately; the worker posts MINE_PROGRESS
        per file, then MINE_DONE (or MINE_ERROR).
        """
        rid = self._next_rid()
        handle = MineHandle()
        self._post(BackendEvent(
            type=EventType.MINE_STARTED,
            payload={"source_dir": source_dir, "mode": mode},
            request_id=rid,
        ))
        self._work_q.put(_WorkItem(
            kind="mine",
            kwargs=dict(
                source_dir=source_dir,
                mode=mode,
                wing=wing,
                agent=agent,
                limit=limit,
                dry_run=dry_run,
                respect_gitignore=respect_gitignore,
                extract_mode=extract_mode,
                timeout=timeout,
            ),
            request_id=rid,
            mine_handle=handle,
        ))
        return handle, rid

    def status(self) -> int:
        """Enqueue a status check. Returns request_id.

        Posts STATUS_STARTED immediately; worker posts STATUS_DONE or STATUS_ERROR.
        """
        rid = self._next_rid()
        self._post(BackendEvent(type=EventType.STATUS_STARTED, payload={}, request_id=rid))
        self._work_q.put(_WorkItem(kind="status", request_id=rid))
        return rid

    def start_mcp_server(self, *, stderr_log: Optional[str] = None) -> int:
        """Enqueue MCP server start. Returns request_id."""
        rid = self._next_rid()
        self._work_q.put(_WorkItem(
            kind="mcp_start",
            kwargs=dict(stderr_log=stderr_log),
            request_id=rid,
        ))
        return rid

    def stop_mcp_server(self) -> int:
        """Enqueue MCP server stop. Returns request_id."""
        rid = self._next_rid()
        self._work_q.put(_WorkItem(kind="mcp_stop", request_id=rid))
        return rid

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Drain the work queue, one item at a time, forever (until 'shutdown')."""
        while True:
            try:
                item = self._work_q.get()
                if item.kind == "shutdown":
                    logger.debug("Worker received shutdown sentinel")
                    break
                self._dispatch(item)
            except Exception as e:
                logger.exception("Unhandled exception in BackendController worker")
                self._post(BackendEvent(
                    type=EventType.CONTROLLER_ERROR,
                    payload={"error": str(e)},
                ))

    def _dispatch(self, item: _WorkItem) -> None:
        rid = item.request_id
        try:
            if item.kind == "search":
                result: SearchResult = self._adapter.run_search(**item.kwargs)
                if result.ok:
                    self._post(BackendEvent(type=EventType.SEARCH_DONE, payload=result, request_id=rid))
                else:
                    self._post(BackendEvent(
                        type=EventType.SEARCH_ERROR,
                        payload={"error": result.error},
                        request_id=rid,
                    ))

            elif item.kind == "mine":
                kw = dict(item.kwargs)
                mode = kw.pop("mode", "projects")
                source_dir = kw.pop("source_dir")

                def _on_progress(ev: MineProgressEvent):
                    self._post(BackendEvent(type=EventType.MINE_PROGRESS, payload=ev, request_id=rid))

                if mode == "projects":
                    result: MineResult = self._adapter.run_mine_projects(
                        source_dir,
                        on_progress=_on_progress,
                        handle=item.mine_handle,
                        **{k: v for k, v in kw.items()
                           if k in ("wing", "agent", "limit", "dry_run",
                                    "respect_gitignore", "timeout")},
                    )
                else:
                    result: MineResult = self._adapter.run_mine_convos(
                        source_dir,
                        on_progress=_on_progress,
                        handle=item.mine_handle,
                        **{k: v for k, v in kw.items()
                           if k in ("wing", "agent", "limit", "dry_run",
                                    "extract_mode", "timeout")},
                    )

                if result.ok:
                    self._post(BackendEvent(type=EventType.MINE_DONE, payload=result, request_id=rid))
                else:
                    self._post(BackendEvent(
                        type=EventType.MINE_ERROR,
                        payload={"error": result.error},
                        request_id=rid,
                    ))

            elif item.kind == "status":
                result: PalaceStatus = self._adapter.run_status()
                if result.ok:
                    self._post(BackendEvent(type=EventType.STATUS_DONE, payload=result, request_id=rid))
                else:
                    self._post(BackendEvent(
                        type=EventType.STATUS_ERROR,
                        payload={"error": result.error},
                        request_id=rid,
                    ))

            elif item.kind == "mcp_start":
                mcp_status: McpServerStatus = self._adapter.start_mcp_server(**item.kwargs)
                self._post(BackendEvent(type=EventType.MCP_STARTED, payload=mcp_status, request_id=rid))

            elif item.kind == "mcp_stop":
                mcp_status: McpServerStatus = self._adapter.stop_mcp_server()
                self._post(BackendEvent(type=EventType.MCP_STOPPED, payload=mcp_status, request_id=rid))

        except Exception as e:
            logger.exception("Error dispatching work item kind=%s", item.kind)
            self._post(BackendEvent(
                type=EventType.CONTROLLER_ERROR,
                payload={"error": str(e), "kind": item.kind},
                request_id=rid,
            ))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_rid(self) -> int:
        with self._counter_lock:
            self._request_counter += 1
            return self._request_counter

    def _post(self, event: BackendEvent) -> None:
        """Post event to public queue and call on_event callback."""
        self.events.put(event)
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:
                logger.exception("on_event callback raised; event type=%s", event.type)

    def __repr__(self) -> str:
        return (
            f"BackendController(palace_path={self.palace_path!r}, "
            f"started={self._started}, "
            f"queue_size={self._work_q.qsize()})"
        )
