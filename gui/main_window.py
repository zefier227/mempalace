"""
gui/main_window.py -- MemPalace MVP single-window GUI.
======================================================

One QMainWindow with a QTabWidget that hosts four panels:

  InitPanel    -- set palace path, run safe_init()
  MinePanel    -- pick a project dir, run mine, see live progress
  StatusPanel  -- palace overview (wings / rooms / drawer counts)
  SearchPanel  -- natural-language search with result previews

All heavy work is delegated to QtController (which delegates to
MemPalaceAdapter).  This file contains ZERO business logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Qt, Slot, QSize, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.qt_controller import QtController
from mempalace.gui_adapter import (
    InitResult,
    MineProgressEvent,
    MineResult,
    PalaceStatus,
    SearchResult,
    SearchHit,
    WakeUpResult,
    CompressResult,
    ContextPackResult,
)


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

_MONO = QFont("Menlo, Monaco, Courier New, monospace", 12)


def _label(text: str, bold: bool = False) -> QLabel:
    lbl = QLabel(text)
    if bold:
        f = lbl.font()
        f.setBold(True)
        lbl.setFont(f)
    return lbl


def _hline() -> QWidget:
    """A thin horizontal separator widget."""
    w = QWidget()
    w.setFixedHeight(1)
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    w.setStyleSheet("background: #ddd;")
    return w


# ---------------------------------------------------------------------------
# 1. InitPanel
# ---------------------------------------------------------------------------


class InitPanel(QWidget):
    """Set palace path and call safe_init()."""

    def __init__(self, controller: QtController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)
        root.setSpacing(12)
        root.setContentsMargins(20, 20, 20, 20)

        root.addWidget(_label("Palace Settings", bold=True))
        root.addWidget(_hline())

        # Palace path row
        path_row = QHBoxLayout()
        path_row.addWidget(_label("Palace path:"))
        self._path_edit = QLineEdit(self._ctrl.palace_path)
        self._path_edit.setPlaceholderText("~/.mempalace/palace")
        path_row.addWidget(self._path_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_palace)
        path_row.addWidget(browse_btn)
        root.addLayout(path_row)

        # Optional room detection
        self._auto_detect_cb = QCheckBox("Auto-detect rooms from a project directory")
        root.addWidget(self._auto_detect_cb)

        self._project_row = QHBoxLayout()
        self._project_row.addWidget(_label("Project dir (optional):"))
        self._project_edit = QLineEdit()
        self._project_edit.setPlaceholderText("Optional -- only needed with auto-detect rooms")
        self._project_row.addWidget(self._project_edit, 1)
        proj_browse = QPushButton("Browse...")
        proj_browse.clicked.connect(self._browse_project)
        self._project_row.addWidget(proj_browse)
        root.addLayout(self._project_row)

        self._auto_detect_cb.toggled.connect(self._project_edit.setEnabled)
        self._auto_detect_cb.toggled.connect(proj_browse.setEnabled)
        self._project_edit.setEnabled(False)
        proj_browse.setEnabled(False)

        # Init button
        self._init_btn = QPushButton("Initialise Palace")
        self._init_btn.setFixedHeight(36)
        self._init_btn.clicked.connect(self._do_init)
        root.addWidget(self._init_btn)

        # Status area
        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

        root.addStretch()

        # Wire controller signals
        self._ctrl.init_finished.connect(self._on_init_done)
        self._ctrl.busy_changed.connect(self._on_busy)
        self._ctrl.palace_switched.connect(self._on_palace_switched)

    @Slot(str)
    def _on_palace_switched(self, new_path: str):
        self._path_edit.setText(new_path)

    def _browse_palace(self):
        d = QFileDialog.getExistingDirectory(self, "Select palace directory", str(Path.home()))
        if d:
            self._path_edit.setText(d)

    def _browse_project(self):
        d = QFileDialog.getExistingDirectory(self, "Select project directory", str(Path.home()))
        if d:
            self._project_edit.setText(d)

    def _do_init(self):
        project_dir = None
        if self._auto_detect_cb.isChecked() and self._project_edit.text().strip():
            project_dir = self._project_edit.text().strip()
        palace_path = self._path_edit.text().strip() or None
        self._ctrl.request_init(
            project_dir=project_dir,
            auto_detect_rooms=self._auto_detect_cb.isChecked(),
            palace_path=palace_path,
        )

    @Slot(object)
    def _on_init_done(self, result: InitResult):
        if result.ok:
            self._path_edit.setText(result.palace_path)
            self._status_lbl.setText(
                "Palace initialised.\n"
                f"Path: {result.palace_path}\n\n"
                "Palace is initialised but not yet indexed.\n"
                "Go to the Mine tab and mine a project to create drawers."
            )
            self._status_lbl.setStyleSheet("color: #555;")
        else:
            self._status_lbl.setText(f"Error: {result.error}")
            self._status_lbl.setStyleSheet("color: red;")

    @Slot(bool)
    def _on_busy(self, busy: bool):
        self._init_btn.setEnabled(not busy)


# ---------------------------------------------------------------------------
# 2. MinePanel
# ---------------------------------------------------------------------------


class MinePanel(QWidget):
    """Pick a project directory, mine it, watch live progress."""

    def __init__(self, controller: QtController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)
        root.setSpacing(12)
        root.setContentsMargins(20, 20, 20, 20)

        root.addWidget(_label("Mine a Project", bold=True))
        root.addWidget(_hline())

        # Project dir row
        dir_row = QHBoxLayout()
        dir_row.addWidget(_label("Project directory:"))
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("/path/to/your/project")
        dir_row.addWidget(self._dir_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        root.addLayout(dir_row)

        # Optional wing override
        wing_row = QHBoxLayout()
        wing_row.addWidget(_label("Wing name (optional):"))
        self._wing_edit = QLineEdit()
        self._wing_edit.setPlaceholderText("Defaults to directory name")
        wing_row.addWidget(self._wing_edit, 1)
        root.addLayout(wing_row)

        # Mine button
        self._mine_btn = QPushButton("Mine Project")
        self._mine_btn.setFixedHeight(36)
        self._mine_btn.clicked.connect(self._do_mine)
        root.addWidget(self._mine_btn)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate by default
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        # Live log
        root.addWidget(_label("Progress:"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_MONO)
        self._log.setMinimumHeight(200)
        root.addWidget(self._log)

        # Wire signals
        self._ctrl.mine_progress.connect(self._on_progress)
        self._ctrl.mine_finished.connect(self._on_mine_done)
        self._ctrl.busy_changed.connect(self._on_busy)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select project directory", str(Path.home()))
        if d:
            self._dir_edit.setText(d)

    def _do_mine(self):
        source_dir = self._dir_edit.text().strip()
        if not source_dir:
            QMessageBox.warning(self, "No directory", "Please select a project directory.")
            return
        wing = self._wing_edit.text().strip() or None
        self._log.clear()
        self._log.append(f"Mining {source_dir} ...\n")
        self._ctrl.request_mine(source_dir, wing=wing)

    @Slot(object)
    def _on_progress(self, event: MineProgressEvent):
        if event.type == "file":
            self._log.append(
                f"  [{event.current}/{event.total}] {event.filename}  +{event.drawers} drawer(s)"
            )
            if event.total and event.total > 0:
                self._progress.setRange(0, event.total)
                self._progress.setValue(event.current)
        elif event.type == "done":
            self._log.append(
                "\nDone.\n"
                f"   Files processed : {event.files_processed}\n"
                f"   Files skipped   : {event.files_skipped}\n"
                f"   Drawers filed   : {event.drawers_filed}\n"
                f"   Wing            : {event.wing}"
            )
            if event.rooms:
                self._log.append("   Rooms:")
                for room, count in sorted(event.rooms.items()):
                    self._log.append(f"     {room}: {count}")

    @Slot(object)
    def _on_mine_done(self, result: MineResult):
        self._progress.setVisible(False)
        if not result.ok:
            self._log.append(f"\nMine failed:\n{result.error}")

    @Slot(bool)
    def _on_busy(self, busy: bool):
        self._mine_btn.setEnabled(not busy)
        self._progress.setVisible(busy)
        if busy:
            self._progress.setRange(0, 0)  # indeterminate while starting


# ---------------------------------------------------------------------------
# 3. StatusPanel
# ---------------------------------------------------------------------------


class StatusPanel(QWidget):
    """Palace overview -- wings, rooms, drawer counts."""

    def __init__(self, controller: QtController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)
        root.setSpacing(12)
        root.setContentsMargins(20, 20, 20, 20)

        hdr = QHBoxLayout()
        hdr.addWidget(_label("Palace Status", bold=True))
        hdr.addStretch()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._ctrl.request_status)
        hdr.addWidget(self._refresh_btn)
        root.addLayout(hdr)
        root.addWidget(_hline())

        # Summary labels
        self._palace_lbl = _label("Palace: --")
        self._chroma_lbl = _label("ChromaDB: --")
        self._drawers_lbl = _label("Total drawers: --")
        self._files_lbl = _label("Total files: --")
        root.addWidget(self._palace_lbl)
        root.addWidget(self._chroma_lbl)
        root.addWidget(self._drawers_lbl)
        root.addWidget(self._files_lbl)

        # Empty state
        self._empty_lbl = QLabel(
            "No data in palace yet.\n\n"
            "1. Go to the Init tab and initialise the palace.\n"
            "2. Go to the Mine tab and mine a project directory.\n"
            "3. Come back here to see your indexed content."
        )
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet("color: #888; padding: 30px;")
        self._empty_lbl.setWordWrap(True)
        root.addWidget(self._empty_lbl)

        # Wings tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Wing / Room", "Drawers"])
        self._tree.setColumnWidth(0, 300)
        self._tree.setVisible(False)
        root.addWidget(self._tree)

        root.addStretch()

        # Wire signals
        self._ctrl.status_finished.connect(self._on_status)
        self._ctrl.busy_changed.connect(self._on_busy)

    @Slot(object)
    def _on_status(self, status: PalaceStatus):
        self._palace_lbl.setText(f"Palace: {status.palace_path or '--'}")
        self._chroma_lbl.setText(f"ChromaDB: {status.chromadb_version or '--'}")

        if not status.ok or status.total_drawers == 0:
            self._drawers_lbl.setText("Total drawers: 0")
            self._files_lbl.setText("Total files: 0")
            self._tree.setVisible(False)
            self._empty_lbl.setVisible(True)
            msg = status.error or "Initialised but not yet indexed."
            if not status.ok:
                msg = status.error or "Palace not found -- run Init first."
            self._empty_lbl.setText(msg)
            return

        self._drawers_lbl.setText(f"Total drawers: {status.total_drawers}")
        self._files_lbl.setText(f"Total files: {status.total_files}")
        self._empty_lbl.setVisible(False)
        self._tree.setVisible(True)
        self._tree.clear()

        for wing in status.wings:
            wing_item = QTreeWidgetItem([wing.name, str(wing.total_drawers)])
            wing_item.setExpanded(True)
            f = wing_item.font(0)
            f.setBold(True)
            wing_item.setFont(0, f)
            for room in wing.rooms:
                room_item = QTreeWidgetItem([f"  {room.name}", str(room.drawers)])
                wing_item.addChild(room_item)
            self._tree.addTopLevelItem(wing_item)

    @Slot(bool)
    def _on_busy(self, busy: bool):
        self._refresh_btn.setEnabled(not busy)


# ---------------------------------------------------------------------------
# 4. SearchPanel
# ---------------------------------------------------------------------------


class SearchPanel(QWidget):
    """Natural-language search with verbatim result previews.

    Mirrors CLI ``mempalace search`` behaviour exactly:
    flat hit list, raw similarity, verbatim drawer text in preview.
    No grouping, no excerpts, no explainability, no threshold.
    """

    def __init__(self, controller: QtController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._hits: List[SearchHit] = []
        self._last_query: str = ""
        self._last_wing: Optional[str] = None
        self._last_room: Optional[str] = None
        self._last_n_results: int = 5
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(20, 20, 20, 20)

        root.addWidget(_label("Search", bold=True))
        root.addWidget(_hline())

        # Query row
        query_row = QHBoxLayout()
        self._query_edit = QLineEdit()
        self._query_edit.setPlaceholderText("What do you want to remember?")
        self._query_edit.returnPressed.connect(self._do_search)
        query_row.addWidget(self._query_edit, 1)

        self._search_btn = QPushButton("Search")
        self._search_btn.setFixedHeight(32)
        self._search_btn.clicked.connect(self._do_search)
        query_row.addWidget(self._search_btn)
        root.addLayout(query_row)

        # Wing filter
        filter_row = QHBoxLayout()
        filter_row.addWidget(_label("Wing:"))
        self._wing_edit = QLineEdit()
        self._wing_edit.setPlaceholderText("Optional")
        filter_row.addWidget(self._wing_edit, 1)
        filter_row.addWidget(_label("Room:"))
        self._room_edit = QLineEdit()
        self._room_edit.setPlaceholderText("Optional")
        filter_row.addWidget(self._room_edit, 1)
        root.addLayout(filter_row)

        # Max results row
        max_row = QHBoxLayout()
        max_row.addWidget(_label("Max results:"))
        self._n_results_spin = QSpinBox()
        self._n_results_spin.setRange(1, 500)
        self._n_results_spin.setValue(5)
        self._n_results_spin.setToolTip("Maximum number of search results to return")
        max_row.addWidget(self._n_results_spin)
        max_row.addStretch()
        root.addLayout(max_row)

        # Splitter: results list | preview
        splitter = QSplitter(Qt.Horizontal)

        # Left: result list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._result_count_lbl = _label("No results")
        left_layout.addWidget(self._result_count_lbl)
        self._results_list = QListWidget()
        self._results_list.currentRowChanged.connect(self._on_result_selected)
        left_layout.addWidget(self._results_list)
        self._show_more_btn = QPushButton("Show more...")
        self._show_more_btn.setVisible(False)
        self._show_more_btn.clicked.connect(self._on_show_more)
        left_layout.addWidget(self._show_more_btn)
        splitter.addWidget(left)

        # Right: preview pane
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._preview_header = QLabel("")
        self._preview_header.setWordWrap(True)
        self._preview_header.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #222; "
            "padding: 4px 6px; background: #f0f0f0; border-radius: 3px;"
        )
        right_layout.addWidget(self._preview_header)

        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setFont(_MONO)
        right_layout.addWidget(self._preview)

        self._meta_lbl = QLabel("")
        self._meta_lbl.setWordWrap(True)
        self._meta_lbl.setStyleSheet("color: #666; font-size: 11px;")
        right_layout.addWidget(self._meta_lbl)

        splitter.addWidget(right)
        splitter.setSizes([300, 500])
        root.addWidget(splitter, 1)

        # Empty state
        self._empty_lbl = QLabel(
            "Type a query above and press Search.\n\n"
            "No results? Make sure the palace has been mined first."
        )
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setStyleSheet("color: #888; padding: 30px;")
        self._empty_lbl.setWordWrap(True)
        root.addWidget(self._empty_lbl)

        # Wire signals
        self._ctrl.search_finished.connect(self._on_search_done)
        self._ctrl.busy_changed.connect(self._on_busy)
        self._ctrl.palace_switched.connect(self._on_palace_switched)

    def _on_palace_switched(self, new_path: str):
        self._results_list.clear()
        self._preview.clear()
        self._preview_header.clear()
        self._meta_lbl.clear()
        self._hits = []
        self._result_count_lbl.setText("No results")
        self._show_more_btn.setVisible(False)
        self._empty_lbl.setVisible(True)

    def _do_search(self):
        q = self._query_edit.text().strip()
        if not q:
            return
        wing = self._wing_edit.text().strip() or None
        room = self._room_edit.text().strip() or None
        n = self._n_results_spin.value()
        self._last_query = q
        self._last_wing = wing
        self._last_room = room
        self._last_n_results = n
        self._ctrl.request_search(q, wing=wing, room=room, n_results=n)

    def _on_show_more(self):
        if not self._last_query:
            return
        self._last_n_results = min(self._last_n_results + 5, 500)
        self._n_results_spin.setValue(self._last_n_results)
        self._ctrl.request_search(
            self._last_query,
            wing=self._last_wing,
            room=self._last_room,
            n_results=self._last_n_results,
        )

    @Slot(object)
    def _on_search_done(self, result: SearchResult):
        self._results_list.clear()
        self._preview.clear()
        self._preview_header.clear()
        self._meta_lbl.clear()
        self._hits = []

        if not result.ok:
            self._result_count_lbl.setText("Search error")
            self._show_more_btn.setVisible(False)
            self._empty_lbl.setText(
                f"Error: {result.error}\n\n"
                "Make sure the palace has been initialised and mined.\n"
                "Check the Init and Mine tabs first."
            )
            self._empty_lbl.setVisible(True)
            return

        self._hits = result.hits
        total = len(self._hits)

        if total == 0:
            self._result_count_lbl.setText("No results")
            self._show_more_btn.setVisible(False)
            self._empty_lbl.setText(
                f'No results for "{result.query}".\n\n'
                "Try different keywords, or check that the relevant project has been mined."
            )
            self._empty_lbl.setVisible(True)
            return

        self._result_count_lbl.setText(f"{total} result{'s' if total != 1 else ''}")
        self._show_more_btn.setVisible(False)
        self._empty_lbl.setVisible(False)

        for i, hit in enumerate(self._hits):
            sim_str = f"{hit.similarity:.3f}" if hit.similarity is not None else ""
            line1 = f"[{i + 1}]  {hit.wing} / {hit.room}  ·  {hit.source_file}  ·  sim {sim_str}"
            first_line = hit.text.strip().split("\n")[0][:70]
            line2 = f"       {first_line}"
            item = QListWidgetItem(f"{line1}\n{line2}")
            item.setData(Qt.UserRole, i)
            self._results_list.addItem(item)

        self._results_list.setCurrentRow(0)

    @Slot(int)
    def _on_result_selected(self, row: int):
        if row < 0 or row >= len(self._hits):
            return
        hit = self._hits[row]
        self._preview.setPlainText(hit.text)
        header = f"{hit.wing} / {hit.room}  ·  {hit.source_file}"
        self._preview_header.setText(header)
        meta_parts = [
            f"Wing: {hit.wing}",
            f"Room: {hit.room}",
            f"Source: {hit.source_file}",
            f"Similarity: {hit.similarity:.3f}",
        ]
        self._meta_lbl.setText("  |  ".join(meta_parts))

    @Slot(bool)
    def _on_busy(self, busy: bool):
        self._search_btn.setEnabled(not busy)
        self._query_edit.setEnabled(not busy)


# ---------------------------------------------------------------------------
# 5. ContextPackPanel
# ---------------------------------------------------------------------------


class WakeUpPanel(QWidget):
    """Wake-up: raw L0+L1 text, mirrors CLI ``mempalace wake-up``."""

    def __init__(self, controller: QtController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)
        root.setSpacing(10)
        root.setContentsMargins(20, 20, 20, 20)

        root.addWidget(_label("Wake-up", bold=True))
        root.addWidget(_hline())

        desc = QLabel(
            "L0 (identity) + L1 (essential story) — same as ``mempalace wake-up`` in the terminal."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; margin-bottom: 6px;")
        root.addWidget(desc)

        wing_row = QHBoxLayout()
        wing_row.addWidget(_label("Wing (optional):"))
        self._wing_edit = QLineEdit()
        self._wing_edit.setPlaceholderText("Leave blank for all wings")
        wing_row.addWidget(self._wing_edit, 1)
        root.addLayout(wing_row)

        btn_row = QHBoxLayout()
        self._gen_btn = QPushButton("Generate wake-up")
        self._gen_btn.setFixedHeight(36)
        self._gen_btn.clicked.connect(self._do_wakeup)
        btn_row.addWidget(self._gen_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(_MONO)
        self._output.setMinimumHeight(200)
        root.addWidget(self._output)

        meta_row = QHBoxLayout()
        self._tokens_lbl = _label("")
        self._tokens_lbl.setStyleSheet("color: #666; font-size: 11px;")
        meta_row.addWidget(self._tokens_lbl)
        meta_row.addStretch()
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setFixedHeight(28)
        self._copy_btn.clicked.connect(self._copy_output)
        meta_row.addWidget(self._copy_btn)
        root.addLayout(meta_row)

        root.addStretch()

        self._ctrl.wakeup_finished.connect(self._on_wakeup_done)
        self._ctrl.busy_changed.connect(self._on_busy)

    def _do_wakeup(self):
        wing = self._wing_edit.text().strip() or None
        self._ctrl.request_wakeup(wing=wing)

    @Slot(object)
    def _on_wakeup_done(self, result: WakeUpResult):
        if not result.ok:
            self._output.setPlainText(f"Error: {result.error}")
            self._tokens_lbl.setText("")
            return
        header = f"Wake-up text (~{result.tokens_est} tokens):\n{'=' * 50}\n"
        self._output.setPlainText(header + result.text)
        self._tokens_lbl.setText(f"~{result.tokens_est} tokens")

    def _copy_output(self):
        from PySide6.QtGui import QGuiApplication

        cb = QGuiApplication.clipboard()
        cb.setText(self._output.toPlainText())

    @Slot(bool)
    def _on_busy(self, busy: bool):
        self._gen_btn.setEnabled(not busy)


class CompressPanel(QWidget):
    """Compress: raw AAAK output, mirrors CLI ``mempalace compress``."""

    def __init__(self, controller: QtController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)
        root.setSpacing(10)
        root.setContentsMargins(20, 20, 20, 20)

        root.addWidget(_label("Compress", bold=True))
        root.addWidget(_hline())

        desc = QLabel("AAAK Dialect compression — same as ``mempalace compress`` in the terminal.")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; margin-bottom: 6px;")
        root.addWidget(desc)

        wing_row = QHBoxLayout()
        wing_row.addWidget(_label("Wing (optional):"))
        self._wing_edit = QLineEdit()
        self._wing_edit.setPlaceholderText("Leave blank for all wings")
        wing_row.addWidget(self._wing_edit, 1)
        root.addLayout(wing_row)

        opt_row = QHBoxLayout()
        self._dry_run_cb = QCheckBox("Dry run (preview only, nothing stored)")
        opt_row.addWidget(self._dry_run_cb)
        opt_row.addStretch()
        root.addLayout(opt_row)

        btn_row = QHBoxLayout()
        self._gen_btn = QPushButton("Compress")
        self._gen_btn.setFixedHeight(36)
        self._gen_btn.clicked.connect(self._do_compress)
        btn_row.addWidget(self._gen_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(_MONO)
        self._output.setMinimumHeight(200)
        root.addWidget(self._output)

        meta_row = QHBoxLayout()
        self._stats_lbl = _label("")
        self._stats_lbl.setStyleSheet("color: #666; font-size: 11px;")
        meta_row.addWidget(self._stats_lbl)
        meta_row.addStretch()
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setFixedHeight(28)
        self._copy_btn.clicked.connect(self._copy_output)
        meta_row.addWidget(self._copy_btn)
        root.addLayout(meta_row)

        root.addStretch()

        self._ctrl.compress_finished.connect(self._on_compress_done)
        self._ctrl.busy_changed.connect(self._on_busy)

    def _do_compress(self):
        wing = self._wing_edit.text().strip() or None
        dry_run = self._dry_run_cb.isChecked()
        self._ctrl.request_compress(wing=wing, dry_run=dry_run)

    @Slot(object)
    def _on_compress_done(self, result: CompressResult):
        if not result.ok:
            self._output.setPlainText(f"Error: {result.error}")
            self._stats_lbl.setText("")
            return
        self._output.setPlainText(result.output)
        if result.drawer_count > 0:
            self._stats_lbl.setText(
                f"{result.drawer_count} drawers | "
                f"{result.orig_tokens_est:,}t → {result.comp_tokens_est:,}t | "
                f"{result.compression_ratio:.1f}x"
            )
        else:
            self._stats_lbl.setText("")

    def _copy_output(self):
        from PySide6.QtGui import QGuiApplication

        cb = QGuiApplication.clipboard()
        cb.setText(self._output.toPlainText())

    @Slot(bool)
    def _on_busy(self, busy: bool):
        self._gen_btn.setEnabled(not busy)


class ContextPackPanel(QWidget):
    """Generate a Context Pack from raw text or a file."""

    def __init__(self, controller: QtController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self._last_result: Optional[ContextPackResult] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)
        root.setSpacing(10)
        root.setContentsMargins(20, 20, 20, 20)

        root.addWidget(_label("Context Pack", bold=True))
        root.addWidget(_hline())

        desc = QLabel(
            "Generate derived artifacts from a text or dialogue.\n"
            "Original text always remains the source of truth."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; margin-bottom: 6px;")
        root.addWidget(desc)

        title_row = QHBoxLayout()
        title_row.addWidget(_label("Title:"))
        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("Optional title")
        title_row.addWidget(self._title_edit, 1)
        root.addLayout(title_row)

        source_row = QHBoxLayout()
        source_row.addWidget(_label("Source:"))
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText("Optional source identifier")
        source_row.addWidget(self._source_edit, 1)
        root.addLayout(source_row)

        wing_room_row = QHBoxLayout()
        wing_room_row.addWidget(_label("Wing:"))
        self._wing_edit = QLineEdit()
        self._wing_edit.setPlaceholderText("Optional")
        wing_room_row.addWidget(self._wing_edit, 1)
        wing_room_row.addWidget(_label("Room:"))
        self._room_edit = QLineEdit()
        self._room_edit.setPlaceholderText("Optional")
        wing_room_row.addWidget(self._room_edit, 1)
        root.addLayout(wing_room_row)

        file_row = QHBoxLayout()
        file_row.addWidget(_label("File:"))
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("Optional -- read text from a file instead")
        file_row.addWidget(self._file_edit, 1)
        file_browse = QPushButton("Browse...")
        file_browse.clicked.connect(self._browse_file)
        file_row.addWidget(file_browse)
        root.addLayout(file_row)

        root.addWidget(_label("Text (or load from file above):"))
        self._text_edit = QTextEdit()
        self._text_edit.setPlaceholderText("Paste your text or dialogue here...")
        self._text_edit.setMinimumHeight(120)
        self._text_edit.setFont(_MONO)
        root.addWidget(self._text_edit)

        btn_row = QHBoxLayout()
        self._gen_btn = QPushButton("Generate Context Pack")
        self._gen_btn.setFixedHeight(36)
        self._gen_btn.clicked.connect(self._do_generate)
        btn_row.addWidget(self._gen_btn)
        self._llm_cb = QCheckBox("Use LLM (BYO-LLM)")
        self._llm_cb.setToolTip("Requires LLM_ENDPOINT and LLM_MODEL env vars")
        btn_row.addWidget(self._llm_cb)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._result_tabs = QTabWidget()
        self._result_tabs.setVisible(False)

        self._recap_edit = QTextEdit()
        self._recap_edit.setReadOnly(True)
        self._recap_edit.setFont(_MONO)
        recap_copy = QPushButton("Copy")
        recap_copy.clicked.connect(lambda: self._copy_to_clipboard(self._recap_edit.toPlainText()))
        recap_layout = QVBoxLayout()
        recap_header = QHBoxLayout()
        recap_header.addWidget(_label("Detailed Recap"))
        recap_header.addStretch()
        recap_header.addWidget(recap_copy)
        recap_layout.addLayout(recap_header)
        recap_layout.addWidget(self._recap_edit)
        recap_w = QWidget()
        recap_w.setLayout(recap_layout)
        self._result_tabs.addTab(recap_w, "Recap")

        self._wakeup_edit = QTextEdit()
        self._wakeup_edit.setReadOnly(True)
        self._wakeup_edit.setFont(_MONO)
        wakeup_copy = QPushButton("Copy")
        wakeup_copy.clicked.connect(
            lambda: self._copy_to_clipboard(self._wakeup_edit.toPlainText())
        )
        wakeup_layout = QVBoxLayout()
        wakeup_header = QHBoxLayout()
        wakeup_header.addWidget(_label("Wake-up"))
        wakeup_header.addStretch()
        wakeup_header.addWidget(wakeup_copy)
        wakeup_layout.addLayout(wakeup_header)
        wakeup_layout.addWidget(self._wakeup_edit)
        wakeup_w = QWidget()
        wakeup_w.setLayout(wakeup_layout)
        self._result_tabs.addTab(wakeup_w, "Wake-up")

        self._aaak_edit = QTextEdit()
        self._aaak_edit.setReadOnly(True)
        self._aaak_edit.setFont(_MONO)
        aaak_copy = QPushButton("Copy")
        aaak_copy.clicked.connect(lambda: self._copy_to_clipboard(self._aaak_edit.toPlainText()))
        aaak_layout = QVBoxLayout()
        aaak_header = QHBoxLayout()
        aaak_header.addWidget(_label("AAAK Compressed"))
        aaak_header.addStretch()
        aaak_header.addWidget(aaak_copy)
        aaak_layout.addLayout(aaak_header)
        aaak_layout.addWidget(self._aaak_edit)
        aaak_w = QWidget()
        aaak_w.setLayout(aaak_layout)
        self._result_tabs.addTab(aaak_w, "AAAK")

        self._prompt_edit = QTextEdit()
        self._prompt_edit.setReadOnly(True)
        self._prompt_edit.setFont(_MONO)
        prompt_copy = QPushButton("Copy")
        prompt_copy.clicked.connect(
            lambda: self._copy_to_clipboard(self._prompt_edit.toPlainText())
        )
        prompt_layout = QVBoxLayout()
        prompt_header = QHBoxLayout()
        prompt_header.addWidget(_label("Reusable Prompt"))
        prompt_header.addStretch()
        prompt_header.addWidget(prompt_copy)
        prompt_layout.addLayout(prompt_header)
        prompt_layout.addWidget(self._prompt_edit)
        prompt_w = QWidget()
        prompt_w.setLayout(prompt_layout)
        self._result_tabs.addTab(prompt_w, "Prompt")

        root.addWidget(self._result_tabs)

        self._stats_lbl = QLabel("")
        self._stats_lbl.setStyleSheet("color: #666; font-size: 11px;")
        self._stats_lbl.setVisible(False)
        root.addWidget(self._stats_lbl)

        self._save_btn = QPushButton("Save to Palace")
        self._save_btn.setFixedHeight(32)
        self._save_btn.setVisible(False)
        self._save_btn.clicked.connect(self._do_save)
        root.addWidget(self._save_btn)

        root.addStretch()

        self._ctrl.context_pack_finished.connect(self._on_result)
        self._ctrl.busy_changed.connect(self._on_busy)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select text file",
            str(Path.home()),
            "Text Files (*.txt *.md *.json *.jsonl);;All Files (*)",
        )
        if path:
            self._file_edit.setText(path)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    self._text_edit.setPlainText(f.read())
            except Exception as e:
                self._text_edit.setPlainText(f"Error reading file: {e}")

    def _do_generate(self):
        raw_text = self._text_edit.toPlainText().strip()
        if not raw_text:
            QMessageBox.warning(self, "No text", "Please paste text or load a file.")
            return
        self._ctrl.request_context_pack(
            raw_text=raw_text,
            title=self._title_edit.text().strip(),
            source=self._source_edit.text().strip() or self._file_edit.text().strip(),
            wing=self._wing_edit.text().strip(),
            room=self._room_edit.text().strip(),
            use_llm=self._llm_cb.isChecked(),
        )

    @Slot(object)
    def _on_result(self, result: ContextPackResult):
        if not result.ok:
            QMessageBox.warning(self, "Error", result.error or "Context Pack generation failed.")
            return

        self._last_result = result
        self._recap_edit.setPlainText(result.detailed_recap)
        self._wakeup_edit.setPlainText(result.wake_up)
        self._aaak_edit.setPlainText(result.aaak_text)
        self._prompt_edit.setPlainText(result.reusable_prompt)
        self._result_tabs.setVisible(True)

        self._stats_lbl.setText(
            f"Original: ~{result.original_tokens_est}t | "
            f"Recap: ~{result.recap_tokens_est}t ({result.recap_method}) | "
            f"Wake-up: ~{result.wakeup_tokens_est}t | "
            f"AAAK: ~{result.aaak_tokens_est}t | "
            f"Prompt: ~{result.prompt_tokens_est}t ({result.prompt_method})"
        )
        self._stats_lbl.setVisible(True)
        self._save_btn.setVisible(True)

    def _do_save(self):
        if not self._last_result:
            return
        result = self._ctrl.save_context_pack(self._last_result)
        filed = result.get("filed", 0)
        QMessageBox.information(
            self,
            "Saved",
            f"Saved {filed} artifacts to palace.\n"
            f"Wing: {result.get('wing', '')}, Room: {result.get('room', '')}",
        )

    def _copy_to_clipboard(self, text: str):
        from PySide6.QtGui import QGuiApplication

        cb = QGuiApplication.clipboard()
        cb.setText(text)

    @Slot(bool)
    def _on_busy(self, busy: bool):
        self._gen_btn.setEnabled(not busy)


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Single-window MemPalace MVP."""

    def __init__(self, controller: QtController, parent=None):
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle(f"MemPalace — {controller.palace_path}")
        self.setMinimumSize(QSize(900, 650))
        self._build_ui()
        self._wire_signals()
        # Defer the initial status refresh until the event loop starts,
        # so the window is fully constructed before any I/O begins.
        QTimer.singleShot(0, self._ctrl.request_status)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._init_panel = InitPanel(self._ctrl)
        self._mine_panel = MinePanel(self._ctrl)
        self._status_panel = StatusPanel(self._ctrl)
        self._search_panel = SearchPanel(self._ctrl)
        self._wakeup_panel = WakeUpPanel(self._ctrl)
        self._compress_panel = CompressPanel(self._ctrl)
        self._cp_panel = ContextPackPanel(self._ctrl)

        self._tabs.addTab(self._init_panel, "Init")
        self._tabs.addTab(self._mine_panel, "Mine")
        self._tabs.addTab(self._status_panel, "Status")
        self._tabs.addTab(self._search_panel, "Search")
        self._tabs.addTab(self._wakeup_panel, "Wake-up")
        self._tabs.addTab(self._compress_panel, "Compress")
        self._tabs.addTab(self._cp_panel, "Context Pack")
        layout.addWidget(self._tabs)

        # Status bar at bottom
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage(f"Palace: {self._ctrl.palace_path}")

    def _wire_signals(self):
        self._ctrl.error.connect(self._show_error)
        self._ctrl.busy_changed.connect(self._on_busy)
        self._ctrl.mine_finished.connect(self._on_mine_finished)
        self._ctrl.palace_switched.connect(self._on_palace_switched)

    @Slot(str)
    def _on_palace_switched(self, new_path: str):
        self.setWindowTitle(f"MemPalace — {new_path}")
        self._statusbar.showMessage(f"Palace: {new_path}")

    @Slot(str)
    def _show_error(self, msg: str):
        self._statusbar.showMessage(f"Error: {msg}", 8000)

    @Slot(bool)
    def _on_busy(self, busy: bool):
        verb = "Working..." if busy else f"Palace: {self._ctrl.palace_path}"
        self._statusbar.showMessage(verb)

    @Slot(object)
    def _on_mine_finished(self, result: MineResult):
        if result.ok:
            # Auto-switch to Status tab after a successful mine
            self._tabs.setCurrentWidget(self._status_panel)
            self._statusbar.showMessage("Mine complete -- status updated.", 5000)
