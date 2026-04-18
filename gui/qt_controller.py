"""
gui/qt_controller.py — Qt-side controller for the MemPalace GUI.
================================================================

Thin layer between PySide6 widgets and ``MemPalaceAdapter``.

Responsibilities
----------------
* Owns exactly ONE ``MemPalaceAdapter`` instance (one palace path per session).
* Runs every blocking adapter call on a ``QThread`` worker so the Qt event
  loop (and therefore the UI) is never blocked.
* Emits typed Qt signals that the MainWindow connects to — no business logic
  lives in the widget layer.
* Serialises search/status calls: a ``busy`` flag prevents a second call from
  starting while one is in progress (mine excluded — it manages its own thread).

Signal contract
---------------
init_finished(InitResult)
    safe_init() completed.  result.ok tells the UI whether it succeeded.

mine_progress(MineProgressEvent)
    One progress update (file processed or final "done" summary) from mine.

mine_finished(MineResult)
    mine subprocess exited.  On success, the controller auto-triggers a
    status refresh so the StatusPanel updates without a user click.

status_finished(PalaceStatus)
    run_status() completed.

search_finished(SearchResult)
    run_search() completed.

busy_changed(bool)
    Emitted when a blocking operation starts (True) or ends (False).
    Use to enable/disable buttons.

error(str)
    Human-readable error string.  Always accompanied by the relevant
    *_finished signal carrying ok=False, so the UI has one place to
    display errors.

Thread safety
-------------
All signals are emitted from worker QThreads; Qt's queued connection
mechanism delivers them safely to the main thread.  The ``_busy`` flag is
read/written only from the main thread (inside ``_set_busy``), which is
called before/after starting each worker via ``QMetaObject.invokeMethod``
or direct slot connection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer

from mempalace.gui_adapter import (
    MemPalaceAdapter,
    InitResult,
    MineResult,
    PalaceStatus,
    SearchResult,
    WakeUpResult,
    CompressResult,
    ContextPackResult,
)

logger = logging.getLogger("mempalace.gui.controller")


# ---------------------------------------------------------------------------
# Worker base
# ---------------------------------------------------------------------------


class _Worker(QThread):
    """Base class: runs one adapter call, then emits a done signal."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("mempalace-worker")


# ---------------------------------------------------------------------------
# Concrete workers (one per operation)
# ---------------------------------------------------------------------------


class _InitWorker(_Worker):
    finished = Signal(object)  # InitResult

    def __init__(
        self, adapter: MemPalaceAdapter, project_dir: Optional[str], auto_detect: bool, parent=None
    ):
        super().__init__(parent)
        self._adapter = adapter
        self._project_dir = project_dir
        self._auto_detect = auto_detect

    def run(self):
        result = self._adapter.safe_init(
            project_dir=self._project_dir,
            auto_detect_rooms=self._auto_detect,
        )
        self.finished.emit(result)


class _MineWorker(_Worker):
    progress = Signal(object)  # MineProgressEvent
    finished = Signal(object)  # MineResult

    def __init__(
        self, adapter: MemPalaceAdapter, source_dir: str, wing: Optional[str], parent=None
    ):
        super().__init__(parent)
        self._adapter = adapter
        self._source_dir = source_dir
        self._wing = wing

    def run(self):
        result = self._adapter.run_mine_projects(
            self._source_dir,
            wing=self._wing,
            on_progress=self.progress.emit,
        )
        self.finished.emit(result)


class _StatusWorker(_Worker):
    finished = Signal(object)  # PalaceStatus

    def __init__(self, adapter: MemPalaceAdapter, parent=None):
        super().__init__(parent)
        self._adapter = adapter

    def run(self):
        result = self._adapter.run_status()
        self.finished.emit(result)


class _SearchWorker(_Worker):
    finished = Signal(object)  # SearchResult

    def __init__(
        self,
        adapter: MemPalaceAdapter,
        query: str,
        wing: Optional[str],
        room: Optional[str],
        n_results: int,
        parent=None,
    ):
        super().__init__(parent)
        self._adapter = adapter
        self._query = query
        self._wing = wing
        self._room = room
        self._n_results = n_results

    def run(self):
        result = self._adapter.run_search(
            self._query,
            wing=self._wing,
            room=self._room,
            n_results=self._n_results,
        )
        self.finished.emit(result)


class _ContextPackWorker(_Worker):
    finished = Signal(object)  # ContextPackResult

    def __init__(
        self,
        adapter: MemPalaceAdapter,
        raw_text: str,
        title: str,
        source: str,
        wing: str,
        room: str,
        use_llm: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._adapter = adapter
        self._raw_text = raw_text
        self._title = title
        self._source = source
        self._wing = wing
        self._room = room
        self._use_llm = use_llm

    def run(self):
        result = self._adapter.run_context_pack(
            self._raw_text,
            title=self._title,
            source=self._source,
            wing=self._wing,
            room=self._room,
            use_llm=self._use_llm,
        )
        self.finished.emit(result)


class _WakeUpWorker(_Worker):
    finished = Signal(object)

    def __init__(self, adapter: MemPalaceAdapter, wing: Optional[str], parent=None):
        super().__init__(parent)
        self._adapter = adapter
        self._wing = wing

    def run(self):
        result = self._adapter.run_wakeup(wing=self._wing)
        self.finished.emit(result)


class _CompressWorker(_Worker):
    finished = Signal(object)

    def __init__(
        self,
        adapter: MemPalaceAdapter,
        wing: Optional[str],
        dry_run: bool,
        parent=None,
    ):
        super().__init__(parent)
        self._adapter = adapter
        self._wing = wing
        self._dry_run = dry_run

    def run(self):
        result = self._adapter.run_compress(wing=self._wing, dry_run=self._dry_run)
        self.finished.emit(result)


# ---------------------------------------------------------------------------
# QtController
# ---------------------------------------------------------------------------


class QtController(QObject):
    """
    Qt-side execution boundary for MemPalace GUI v1.

    Usage::

        ctrl = QtController(palace_path="~/my_palace")
        ctrl.init_finished.connect(window.on_init_done)
        ctrl.mine_progress.connect(window.on_mine_progress)
        ctrl.mine_finished.connect(window.on_mine_done)
        ctrl.status_finished.connect(window.on_status)
        ctrl.search_finished.connect(window.on_search)
        ctrl.busy_changed.connect(window.set_busy)
        ctrl.error.connect(window.show_error)

        ctrl.request_init()
        ctrl.request_mine("/path/to/project")
        ctrl.request_search("why GraphQL")
        ctrl.request_status()
    """

    # Public signals
    init_finished = Signal(object)  # InitResult
    mine_progress = Signal(object)  # MineProgressEvent
    mine_finished = Signal(object)  # MineResult
    status_finished = Signal(object)  # PalaceStatus
    search_finished = Signal(object)  # SearchResult
    context_pack_finished = Signal(object)  # ContextPackResult
    wakeup_finished = Signal(object)  # WakeUpResult
    compress_finished = Signal(object)  # CompressResult
    busy_changed = Signal(bool)
    error = Signal(str)
    palace_switched = Signal(str)  # new palace path
    navigate_to_wakeup = Signal(str)  # wing
    navigate_to_compress = Signal(str)  # wing

    def __init__(self, palace_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._adapter = MemPalaceAdapter(palace_path=palace_path)
        self._busy = False
        self._mine_worker: Optional[_MineWorker] = None
        self._search_worker: Optional[_SearchWorker] = None
        self._status_worker: Optional[_StatusWorker] = None
        self._init_worker: Optional[_InitWorker] = None
        self._cp_worker: Optional[_ContextPackWorker] = None
        self._wakeup_worker: Optional[_WakeUpWorker] = None
        self._compress_worker: Optional[_CompressWorker] = None

    @property
    def palace_path(self) -> str:
        return self._adapter.palace_path

    @property
    def busy(self) -> bool:
        return self._busy

    # ------------------------------------------------------------------
    # Public request methods (called from main thread)
    # ------------------------------------------------------------------

    def request_init(
        self,
        project_dir: Optional[str] = None,
        auto_detect_rooms: bool = False,
        palace_path: Optional[str] = None,
    ) -> None:
        """Initialise the palace (non-blocking).

        If palace_path is provided the adapter is switched to that path
        before init runs.  This allows the Init panel to change the active
        palace without a separate switch_palace call.
        """
        if self._busy:
            self.error.emit("Another operation is in progress. Please wait.")
            return
        if palace_path and palace_path != self._adapter.palace_path:
            self._adapter.switch_palace(palace_path)
            self.palace_switched.emit(self._adapter.palace_path)
        self._set_busy(True)
        w = _InitWorker(self._adapter, project_dir, auto_detect_rooms, parent=self)
        w.finished.connect(self._on_init_done)
        w.finished.connect(w.deleteLater)
        self._init_worker = w
        w.start()

    def request_mine(
        self,
        source_dir: str,
        wing: Optional[str] = None,
    ) -> None:
        """Start a mine operation (non-blocking). Ignores if busy."""
        if self._busy:
            self.error.emit("Another operation is in progress. Please wait.")
            return
        self._set_busy(True)
        w = _MineWorker(self._adapter, source_dir, wing, parent=self)
        w.progress.connect(self.mine_progress)
        w.finished.connect(self._on_mine_done)
        w.finished.connect(w.deleteLater)
        self._mine_worker = w
        w.start()

    def request_status(self) -> None:
        """Refresh palace status (non-blocking). Ignores if busy."""
        if self._busy:
            return  # silently skip auto-refresh during mine
        self._set_busy(True)
        w = _StatusWorker(self._adapter, parent=self)
        w.finished.connect(self._on_status_done)
        w.finished.connect(w.deleteLater)
        self._status_worker = w
        w.start()

    def request_search(
        self,
        query: str,
        wing: Optional[str] = None,
        room: Optional[str] = None,
        n_results: int = 5,
    ) -> None:
        """Run a search (non-blocking). Ignores if busy."""
        if not query.strip():
            return
        if self._busy:
            self.error.emit("Another operation is in progress. Please wait.")
            return
        self._set_busy(True)
        w = _SearchWorker(self._adapter, query, wing, room, n_results, parent=self)
        w.finished.connect(self._on_search_done)
        w.finished.connect(w.deleteLater)
        self._search_worker = w
        w.start()

    def request_context_pack(
        self,
        raw_text: str,
        title: str = "",
        source: str = "",
        wing: str = "",
        room: str = "",
        use_llm: bool = False,
    ) -> None:
        """Build a Context Pack (non-blocking). Ignores if busy."""
        if not raw_text.strip():
            return
        if self._busy:
            self.error.emit("Another operation is in progress. Please wait.")
            return
        self._set_busy(True)
        w = _ContextPackWorker(
            self._adapter,
            raw_text,
            title,
            source,
            wing,
            room,
            use_llm,
            parent=self,
        )
        w.finished.connect(self._on_context_pack_done)
        w.finished.connect(w.deleteLater)
        self._cp_worker = w
        w.start()

    def save_context_pack(self, cp_result: ContextPackResult) -> dict:
        """Save context pack artifacts to palace (synchronous, fast)."""
        return self._adapter.save_context_pack(cp_result)

    def request_wakeup(self, wing: Optional[str] = None) -> None:
        """Run wake-up (non-blocking). Ignores if busy."""
        if self._busy:
            self.error.emit("Another operation is in progress. Please wait.")
            return
        self._set_busy(True)
        w = _WakeUpWorker(self._adapter, wing, parent=self)
        w.finished.connect(self._on_wakeup_done)
        w.finished.connect(w.deleteLater)
        self._wakeup_worker = w
        w.start()

    def request_compress(
        self,
        wing: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        """Run compress (non-blocking). Ignores if busy."""
        if self._busy:
            self.error.emit("Another operation is in progress. Please wait.")
            return
        self._set_busy(True)
        w = _CompressWorker(self._adapter, wing, dry_run, parent=self)
        w.finished.connect(self._on_compress_done)
        w.finished.connect(w.deleteLater)
        self._compress_worker = w
        w.start()

    # ------------------------------------------------------------------
    # Internal slots (called from worker threads via Qt queued connection)
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_init_done(self, result: InitResult) -> None:
        self._set_busy(False)
        if not result.ok:
            self.error.emit(f"Init failed: {result.error}")
        self.init_finished.emit(result)

    @Slot(object)
    def _on_mine_done(self, result: MineResult) -> None:
        self._set_busy(False)
        if not result.ok:
            self.error.emit(f"Mine failed: {result.error}")
        self.mine_finished.emit(result)
        # Auto-refresh status after a successful mine so the StatusPanel
        # updates without requiring a manual click.
        # Use QTimer.singleShot so the status request is deferred until
        # after the event loop processes mine_finished signals (including
        # the tab switch in MainWindow).  This avoids a busy-flag race
        # where request_status() would set busy=True before the
        # mine_finished handlers have run.
        if result.ok:
            QTimer.singleShot(0, self.request_status)

    @Slot(object)
    def _on_status_done(self, result: PalaceStatus) -> None:
        self._set_busy(False)
        if not result.ok:
            logger.debug("Status returned error: %s", result.error)
            # Don't emit error for status — empty-state is normal before mine
        self.status_finished.emit(result)

    @Slot(object)
    def _on_search_done(self, result: SearchResult) -> None:
        self._set_busy(False)
        if not result.ok:
            self.error.emit(f"Search failed: {result.error}")
        self.search_finished.emit(result)

    @Slot(object)
    def _on_context_pack_done(self, result: ContextPackResult) -> None:
        self._set_busy(False)
        if not result.ok:
            self.error.emit(f"Context Pack failed: {result.error}")
        self.context_pack_finished.emit(result)

    @Slot(object)
    def _on_wakeup_done(self, result: WakeUpResult) -> None:
        self._set_busy(False)
        if not result.ok:
            self.error.emit(f"Wake-up failed: {result.error}")
        self.wakeup_finished.emit(result)

    @Slot(object)
    def _on_compress_done(self, result: CompressResult) -> None:
        self._set_busy(False)
        if not result.ok:
            self.error.emit(f"Compress failed: {result.error}")
        self.compress_finished.emit(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        if self._busy != busy:
            self._busy = busy
            self.busy_changed.emit(busy)

    def chromadb_version_info(self) -> dict:
        """Synchronous utility — cheap, safe to call from main thread."""
        return MemPalaceAdapter.check_chromadb_version()

    def palace_exists(self) -> dict:
        """Synchronous utility — cheap, safe to call from main thread."""
        return MemPalaceAdapter.check_palace_exists(self._adapter.palace_path)

    def switch_palace(self, new_path: str) -> None:
        """Switch to a different palace path (non-blocking safe).

        Creates a new adapter for the new path, invalidates the old chroma
        client, and emits palace_switched so panels can reset their state.
        Must NOT be called while busy.
        """
        if self._busy:
            self.error.emit("Cannot switch palace while an operation is running.")
            return
        new_resolved = str(Path(new_path).expanduser().resolve())
        if new_resolved == self._adapter.palace_path:
            return
        self._adapter.switch_palace(new_resolved)
        self.palace_switched.emit(self._adapter.palace_path)
