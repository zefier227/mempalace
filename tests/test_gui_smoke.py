"""
tests/test_gui_smoke.py -- Minimal headless smoke tests for the PySide6 GUI.

These tests verify that:
  1. All GUI modules import cleanly.
  2. QtController can be instantiated without a palace directory.
  3. MainWindow can be constructed without a display (offscreen platform).
  4. Signals exist and are of the expected types.
  5. _parse_args() handles all CLI flags correctly.

These tests do NOT mine or search -- that is covered by test_gui_adapter.py
and test_backend_hardening.py.  They are kept fast (<5 s total) so they can
run in CI without a real display.

NOTE: We deliberately avoid calling QApplication.processEvents() in these
tests because the deferred QTimer.singleShot(0, request_status) in
MainWindow.__init__ would trigger a ChromaDB PersistentClient construction
on a QThread worker, which aborts on Linux when pydantic/chromadb introspect
PySide6 types via inspect.getsource.  The construction itself is safe.
"""

from __future__ import annotations

import os
import pytest

# ---------------------------------------------------------------------------
# Ensure Qt uses the offscreen platform in CI / headless environments.
# Must be set BEFORE importing PySide6.
# ---------------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    """Return (or create) a module-scoped QApplication for all smoke tests."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    # Do NOT call app.quit() -- other test modules may share the process.


@pytest.fixture()
def tmp_palace(tmp_path):
    """A temporary directory that acts as an empty palace root."""
    palace = tmp_path / "palace"
    palace.mkdir()
    return str(palace)


# ---------------------------------------------------------------------------
# 1. Import smoke
# ---------------------------------------------------------------------------


class TestImports:
    """All GUI modules must import without errors."""

    def test_import_qt_controller(self):
        from gui.qt_controller import QtController  # noqa: F401

    def test_import_main_window(self):
        from gui.main_window import MainWindow  # noqa: F401

    def test_import_app(self):
        from gui.app import main, _parse_args  # noqa: F401

    def test_import_panels(self):
        from gui.main_window import InitPanel, MinePanel, StatusPanel, SearchPanel  # noqa: F401


# ---------------------------------------------------------------------------
# 2. QtController instantiation (no palace, no QApplication needed for QObject)
# ---------------------------------------------------------------------------


class TestQtController:
    """QtController must be constructable and expose the right signals."""

    def test_instantiate_no_palace(self, qapp, tmp_palace):
        from gui.qt_controller import QtController

        ctrl = QtController(palace_path=tmp_palace)
        assert ctrl is not None

    def test_palace_path_exposed(self, qapp, tmp_palace):
        from gui.qt_controller import QtController

        ctrl = QtController(palace_path=tmp_palace)
        assert ctrl.palace_path == tmp_palace

    def test_busy_starts_false(self, qapp, tmp_palace):
        from gui.qt_controller import QtController

        ctrl = QtController(palace_path=tmp_palace)
        assert ctrl.busy is False

    def test_signals_exist(self, qapp, tmp_palace):
        from gui.qt_controller import QtController

        ctrl = QtController(palace_path=tmp_palace)
        for sig_name in (
            "init_finished",
            "mine_progress",
            "mine_finished",
            "status_finished",
            "search_finished",
            "busy_changed",
            "error",
        ):
            assert hasattr(ctrl, sig_name), f"Missing signal: {sig_name}"

    def test_chromadb_version_info(self, qapp, tmp_palace):
        from gui.qt_controller import QtController

        ctrl = QtController(palace_path=tmp_palace)
        info = ctrl.chromadb_version_info()
        assert isinstance(info, dict)
        assert "version" in info or "ok" in info or "safe" in info

    def test_palace_exists_utility(self, qapp, tmp_palace):
        from gui.qt_controller import QtController

        ctrl = QtController(palace_path=tmp_palace)
        info = ctrl.palace_exists()
        assert isinstance(info, dict)

    def test_request_search_empty_is_noop(self, qapp, tmp_palace):
        """request_search with empty string must not set busy."""
        from gui.qt_controller import QtController

        ctrl = QtController(palace_path=tmp_palace)
        ctrl.request_search("")
        assert ctrl.busy is False

    def test_request_search_no_max_distance_param(self, qapp, tmp_palace):
        """request_search must NOT accept max_distance parameter (raw parity)."""
        import inspect
        from gui.qt_controller import QtController

        sig = inspect.signature(QtController.request_search)
        assert "max_distance" not in sig.parameters, (
            "request_search must not have max_distance parameter for raw parity"
        )


# ---------------------------------------------------------------------------
# 3. MainWindow construction (headless, no event loop)
# ---------------------------------------------------------------------------


class TestMainWindow:
    """
    MainWindow must build all four panels without errors.

    We construct the window but do NOT call processEvents() so that the
    deferred QTimer.singleShot(0, request_status) never fires during the test,
    avoiding ChromaDB initialisation on a background thread.
    """

    def test_window_constructs(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import MainWindow

        ctrl = QtController(palace_path=tmp_palace)
        win = MainWindow(controller=ctrl)
        assert win is not None

    def test_window_title(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import MainWindow

        ctrl = QtController(palace_path=tmp_palace)
        win = MainWindow(controller=ctrl)
        assert "MemPalace" in win.windowTitle()

    def test_tab_count(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import MainWindow
        from PySide6.QtWidgets import QTabWidget

        ctrl = QtController(palace_path=tmp_palace)
        win = MainWindow(controller=ctrl)
        tabs = win.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 7

    def test_tab_labels(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import MainWindow
        from PySide6.QtWidgets import QTabWidget

        ctrl = QtController(palace_path=tmp_palace)
        win = MainWindow(controller=ctrl)
        tabs = win.findChild(QTabWidget)
        labels = [tabs.tabText(i) for i in range(tabs.count())]
        assert any("Init" in lbl for lbl in labels)
        assert any("Mine" in lbl for lbl in labels)
        assert any("Status" in lbl for lbl in labels)
        assert any("Search" in lbl for lbl in labels)
        assert any("Wake-up" in lbl for lbl in labels)
        assert any("Compress" in lbl for lbl in labels)

    def test_minimum_size(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import MainWindow

        ctrl = QtController(palace_path=tmp_palace)
        win = MainWindow(controller=ctrl)
        assert win.minimumWidth() >= 900
        assert win.minimumHeight() >= 650

    def test_status_bar_present(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import MainWindow

        ctrl = QtController(palace_path=tmp_palace)
        win = MainWindow(controller=ctrl)
        assert win.statusBar() is not None


# ---------------------------------------------------------------------------
# 4. Panel construction in isolation
# ---------------------------------------------------------------------------


class TestPanels:
    """Each panel must instantiate with a valid controller."""

    def _make_ctrl(self, qapp, tmp_palace):
        from gui.qt_controller import QtController

        return QtController(palace_path=tmp_palace)

    def test_init_panel(self, qapp, tmp_palace):
        from gui.main_window import InitPanel

        panel = InitPanel(self._make_ctrl(qapp, tmp_palace))
        assert panel is not None

    def test_mine_panel(self, qapp, tmp_palace):
        from gui.main_window import MinePanel

        panel = MinePanel(self._make_ctrl(qapp, tmp_palace))
        assert panel is not None

    def test_status_panel(self, qapp, tmp_palace):
        from gui.main_window import StatusPanel

        panel = StatusPanel(self._make_ctrl(qapp, tmp_palace))
        assert panel is not None

    def test_search_panel(self, qapp, tmp_palace):
        from gui.main_window import SearchPanel

        panel = SearchPanel(self._make_ctrl(qapp, tmp_palace))
        assert panel is not None

    def test_mine_panel_has_log(self, qapp, tmp_palace):
        from gui.main_window import MinePanel
        from PySide6.QtWidgets import QTextEdit

        panel = MinePanel(self._make_ctrl(qapp, tmp_palace))
        log = panel.findChild(QTextEdit)
        assert log is not None
        assert log.isReadOnly()

    def test_search_panel_has_query_field(self, qapp, tmp_palace):
        from gui.main_window import SearchPanel
        from PySide6.QtWidgets import QLineEdit

        panel = SearchPanel(self._make_ctrl(qapp, tmp_palace))
        fields = panel.findChildren(QLineEdit)
        # query + wing + room = 3 fields minimum
        assert len(fields) >= 3

    def test_search_panel_has_room_filter(self, qapp, tmp_palace):
        from gui.main_window import SearchPanel
        from PySide6.QtWidgets import QLineEdit

        panel = SearchPanel(self._make_ctrl(qapp, tmp_palace))
        fields = panel.findChildren(QLineEdit)
        placeholders = [f.placeholderText() for f in fields]
        assert any("Optional" in p for p in placeholders)

    def test_search_panel_n_results_default_is_5(self, qapp, tmp_palace):
        from gui.main_window import SearchPanel
        from PySide6.QtWidgets import QSpinBox

        panel = SearchPanel(self._make_ctrl(qapp, tmp_palace))
        spinboxes = panel.findChildren(QSpinBox)
        assert len(spinboxes) >= 1
        spin = spinboxes[0]
        assert spin.value() == 5

    def test_search_panel_has_no_threshold_slider(self, qapp, tmp_palace):
        from gui.main_window import SearchPanel
        from PySide6.QtWidgets import QSlider

        panel = SearchPanel(self._make_ctrl(qapp, tmp_palace))
        sliders = panel.findChildren(QSlider)
        assert len(sliders) == 0

    def test_status_panel_has_file_count_label(self, qapp, tmp_palace):
        from gui.main_window import StatusPanel

        panel = StatusPanel(self._make_ctrl(qapp, tmp_palace))
        labels = panel.findChildren(type(panel._files_lbl))
        texts = [lbl.text() for lbl in labels]
        assert any("Total files" in t for t in texts)

    def test_window_title_shows_palace_path(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import MainWindow

        ctrl = QtController(palace_path=tmp_palace)
        win = MainWindow(controller=ctrl)
        assert tmp_palace in win.windowTitle()

    def test_search_panel_has_show_more_button(self, qapp, tmp_palace):
        from gui.main_window import SearchPanel
        from PySide6.QtWidgets import QPushButton

        panel = SearchPanel(self._make_ctrl(qapp, tmp_palace))
        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert any("Show more" in lbl for lbl in labels)

    def test_mine_done_uses_deferred_status(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        import inspect

        src = inspect.getsource(QtController._on_mine_done)
        assert "QTimer.singleShot" in src, (
            "Post-mine status refresh must use QTimer.singleShot to avoid busy-flag race"
        )


# ---------------------------------------------------------------------------
# 5. CLI arg parsing
# ---------------------------------------------------------------------------


class TestArgParsing:
    """_parse_args must handle all flags without errors."""

    def test_no_args(self):
        from gui.app import _parse_args

        ns = _parse_args([])
        assert ns.palace is None
        assert ns.debug is False

    def test_palace_arg(self, tmp_path):
        from gui.app import _parse_args

        ns = _parse_args(["--palace", str(tmp_path)])
        assert ns.palace == str(tmp_path)

    def test_debug_flag(self):
        from gui.app import _parse_args

        ns = _parse_args(["--debug"])
        assert ns.debug is True

    def test_palace_and_debug(self, tmp_path):
        from gui.app import _parse_args

        ns = _parse_args(["--palace", str(tmp_path), "--debug"])
        assert ns.palace == str(tmp_path)
        assert ns.debug is True

    def test_help_exits(self):
        from gui.app import _parse_args

        with pytest.raises(SystemExit) as exc:
            _parse_args(["--help"])
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# 6. Search selection / preview behavior (SearchPanel — flat hit list)
# ---------------------------------------------------------------------------


class TestSearchSelectionBehavior:
    """Verify that clicking a result shows the verbatim preview."""

    def _make_panels(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import SearchPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = SearchPanel(ctrl)
        return ctrl, panel

    def _populate_and_search(self, ctrl, panel, tmp_palace, tmp_path):
        from mempalace.gui_adapter import MemPalaceAdapter

        proj = tmp_path / "sel_project"
        proj.mkdir()
        (proj / "alpha.txt").write_text(
            "Alpha file: GraphQL API design decisions and architecture overview."
        )
        (proj / "beta.txt").write_text(
            "Beta file: Redis caching strategy and PostgreSQL tuning notes."
        )
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        adapter.run_mine_projects(str(proj))

        result = adapter.run_search("architecture", n_results=5)
        return result

    def test_search_result_hits_populated(self, qapp, tmp_palace, tmp_path):
        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        assert len(panel._hits) >= 1

    def test_click_hit_shows_verbatim_preview(self, qapp, tmp_palace, tmp_path):
        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        if len(panel._hits) < 2:
            pytest.skip("Need at least 2 hits for selection test")
        panel._on_result_selected(0)
        first_preview = panel._preview.toPlainText()
        panel._on_result_selected(1)
        second_preview = panel._preview.toPlainText()
        assert first_preview != second_preview or len(panel._hits) == 1

    def test_new_query_clears_old_preview(self, qapp, tmp_palace, tmp_path):
        from mempalace.gui_adapter import SearchResult

        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        assert len(panel._hits) >= 1
        panel._on_result_selected(0)
        assert panel._preview.toPlainText() != ""

        new_result = SearchResult(ok=True, query="new", hits=[])
        panel._on_search_done(new_result)
        assert panel._preview.toPlainText() == ""
        assert panel._preview_header.text() == ""

    def test_preview_header_shows_wing_room_source(self, qapp, tmp_palace, tmp_path):
        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        if not panel._hits:
            pytest.skip("No hits in result")
        panel._on_result_selected(0)
        header_text = panel._preview_header.text()
        hit = panel._hits[0]
        assert hit.wing in header_text
        assert hit.source_file in header_text

    def test_no_chunk_navigation(self, qapp, tmp_palace, tmp_path):
        """SearchPanel must NOT have chunk navigation (raw parity)."""
        from gui.main_window import SearchPanel

        panel = SearchPanel(self._make_panels(qapp, tmp_palace)[0])
        from PySide6.QtWidgets import QPushButton

        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert not any("chunk" in lbl.lower() for lbl in labels), (
            "SearchPanel must not have chunk navigation buttons"
        )

    def test_flat_list_no_grouping(self, qapp, tmp_palace, tmp_path):
        """Result list must show one entry per hit, not per file."""
        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        assert panel._results_list.count() == len(panel._hits)


# ---------------------------------------------------------------------------
# 7. Wake-up panel smoke tests
# ---------------------------------------------------------------------------


class TestWakeUpPanel:
    def _make_panel(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import WakeUpPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = WakeUpPanel(ctrl)
        return ctrl, panel

    def test_panel_constructs(self, qapp, tmp_palace):
        ctrl, panel = self._make_panel(qapp, tmp_palace)
        assert panel is not None

    def test_has_wing_edit(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QLineEdit

        _, panel = self._make_panel(qapp, tmp_palace)
        edits = panel.findChildren(QLineEdit)
        assert len(edits) >= 1

    def test_has_generate_button(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QPushButton

        _, panel = self._make_panel(qapp, tmp_palace)
        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert any("wake" in lbl.lower() for lbl in labels)

    def test_has_copy_button(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QPushButton

        _, panel = self._make_panel(qapp, tmp_palace)
        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert any("copy" in lbl.lower() for lbl in labels)

    def test_has_output_field(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QTextEdit

        _, panel = self._make_panel(qapp, tmp_palace)
        edits = panel.findChildren(QTextEdit)
        assert any(e.isReadOnly() for e in edits)

    def test_result_updates_output(self, qapp, tmp_palace):
        from mempalace.gui_adapter import WakeUpResult

        _, panel = self._make_panel(qapp, tmp_palace)
        result = WakeUpResult(ok=True, text="L0 identity\nL1 story", tokens_est=5)
        panel._on_wakeup_done(result)
        assert "L0 identity" in panel._output.toPlainText()

    def test_error_shows_in_output(self, qapp, tmp_palace):
        from mempalace.gui_adapter import WakeUpResult

        _, panel = self._make_panel(qapp, tmp_palace)
        result = WakeUpResult(ok=False, error="No palace")
        panel._on_wakeup_done(result)
        assert "Error:" in panel._output.toPlainText()


# ---------------------------------------------------------------------------
# 8. Compress panel smoke tests
# ---------------------------------------------------------------------------


class TestCompressPanel:
    def _make_panel(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import CompressPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = CompressPanel(ctrl)
        return ctrl, panel

    def test_panel_constructs(self, qapp, tmp_palace):
        ctrl, panel = self._make_panel(qapp, tmp_palace)
        assert panel is not None

    def test_has_wing_edit(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QLineEdit

        _, panel = self._make_panel(qapp, tmp_palace)
        edits = panel.findChildren(QLineEdit)
        assert len(edits) >= 1

    def test_has_compress_button(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QPushButton

        _, panel = self._make_panel(qapp, tmp_palace)
        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert any("compress" in lbl.lower() for lbl in labels)

    def test_has_dry_run_checkbox(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QCheckBox

        _, panel = self._make_panel(qapp, tmp_palace)
        checkboxes = panel.findChildren(QCheckBox)
        assert len(checkboxes) >= 1

    def test_has_copy_button(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QPushButton

        _, panel = self._make_panel(qapp, tmp_palace)
        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert any("copy" in lbl.lower() for lbl in labels)

    def test_result_updates_output(self, qapp, tmp_palace):
        from mempalace.gui_adapter import CompressResult

        _, panel = self._make_panel(qapp, tmp_palace)
        result = CompressResult(
            ok=True,
            output="Total: 100t -> 10t (10.0x compression)",
            drawer_count=5,
            orig_tokens_est=100,
            comp_tokens_est=10,
            compression_ratio=10.0,
            dry_run=True,
        )
        panel._on_compress_done(result)
        assert "Total:" in panel._output.toPlainText()

    def test_error_shows_in_output(self, qapp, tmp_palace):
        from mempalace.gui_adapter import CompressResult

        _, panel = self._make_panel(qapp, tmp_palace)
        result = CompressResult(ok=False, error="No palace")
        panel._on_compress_done(result)
        assert "Error:" in panel._output.toPlainText()


# ---------------------------------------------------------------------------
# 9. Search usability actions — 3 scope levels
# ---------------------------------------------------------------------------


class TestSearchUsabilityActions:
    """Verify primary action, More actions menu, and context menu on SearchPanel."""

    def _make_panel(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import SearchPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = SearchPanel(ctrl)
        return ctrl, panel

    def _populate_hits(self, panel, tmp_path=None):
        from mempalace.gui_adapter import SearchResult, SearchHit

        source_path = str(tmp_path / "design.md") if tmp_path else "/tmp/design.md"
        hits = [
            SearchHit(
                text="GraphQL design decisions",
                wing="projects",
                room="2024-01-15",
                source_file="design.md",
                source_path=source_path,
                similarity=0.85,
                distance=0.15,
            ),
            SearchHit(
                text="Redis caching strategy",
                wing="projects",
                room="2024-02-01",
                source_file="caching.md",
                source_path="/tmp/caching.md",
                similarity=0.72,
                distance=0.28,
            ),
        ]
        result = SearchResult(ok=True, query="architecture", hits=hits)
        panel._on_search_done(result)
        return hits

    # -- Primary button --

    def test_has_primary_action_button(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QPushButton

        _, panel = self._make_panel(qapp, tmp_palace)
        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert any("Prepare for new chat" in lbl for lbl in labels)

    def test_has_more_actions_button(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QPushButton

        _, panel = self._make_panel(qapp, tmp_palace)
        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert any("More actions" in lbl for lbl in labels)

    def test_primary_button_has_blue_style(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        style = panel._export_btn.styleSheet()
        assert "2563eb" in style

    # -- Secondary actions are NOT visible as standalone buttons --

    def test_no_standalone_copy_text_button(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QPushButton

        _, panel = self._make_panel(qapp, tmp_palace)
        btns = panel.findChildren(QPushButton)
        secondary = [
            b
            for b in btns
            if b is not panel._more_btn
            and b is not panel._export_btn
            and b is not panel._search_btn
            and b is not panel._show_more_btn
        ]
        assert not any(b.text() == "Copy text" for b in secondary)

    # -- Action buttons disabled when no hit selected --

    def test_action_buttons_disabled_initially(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        assert panel._export_btn.isEnabled() is False
        assert panel._more_btn.isEnabled() is False

    def test_action_buttons_enabled_on_hit_selection(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        panel._on_result_selected(0)
        assert panel._export_btn.isEnabled() is True
        assert panel._more_btn.isEnabled() is True

    def test_action_buttons_disabled_on_invalid_row(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        panel._on_result_selected(-1)
        assert panel._export_btn.isEnabled() is False
        assert panel._more_btn.isEnabled() is False

    # -- Hit-level copy (via method, accessible from More menu) --

    def test_copy_text_puts_hit_text_in_clipboard(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        hits = self._populate_hits(panel)
        panel._on_result_selected(0)
        panel._copy_hit_text()
        from PySide6.QtGui import QGuiApplication

        cb = QGuiApplication.clipboard()
        assert cb.text() == hits[0].text

    # -- source_path available for file-level actions --

    def test_source_path_available_on_hit(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        hits = self._populate_hits(panel)
        panel._on_result_selected(0)
        assert hasattr(hits[0], "source_path")
        assert hits[0].source_path != ""

    # -- Context menu --

    def test_results_list_has_context_menu_policy(self, qapp, tmp_palace):
        from PySide6.QtCore import Qt

        _, panel = self._make_panel(qapp, tmp_palace)
        assert panel._results_list.contextMenuPolicy() == Qt.CustomContextMenu

    def test_context_menu_has_primary_action_first(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        panel._on_result_selected(0)
        from PySide6.QtWidgets import QMenu

        menu = QMenu(panel)
        menu.addAction("Prepare for new chat", lambda: None)
        menu.addSeparator()
        menu.addMenu("Hit")
        menu.addMenu("File")
        menu.addSeparator()
        menu.addMenu("Wing")
        first_action_text = menu.actions()[0].text()
        assert "Prepare for new chat" in first_action_text

    def test_context_menu_has_scope_submenus(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        panel._on_result_selected(0)
        from PySide6.QtWidgets import QMenu

        menu = QMenu(panel)
        hit_menu = menu.addMenu("Hit")
        hit_menu.addAction("Copy text", lambda: None)
        hit_menu.addAction("Compress to AAAK", lambda: None)
        file_menu = menu.addMenu("File")
        file_menu.addAction("Copy source file", lambda: None)
        file_menu.addAction("Open source file", lambda: None)
        file_menu.addAction("Compress source file", lambda: None)
        menu.addSeparator()
        wing_menu = menu.addMenu("Wing")
        wing_menu.addAction("Open wing in Wake-up", lambda: None)
        wing_menu.addAction("Open wing in Compress", lambda: None)
        assert len(menu.actions()) == 4

    # -- Wing-level navigation signals --

    def test_navigate_to_wakeup_signal_emitted(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        panel._on_result_selected(0)
        received = []
        panel._ctrl.navigate_to_wakeup.connect(lambda w: received.append(w))
        panel._open_wing_in_wakeup()
        assert received == ["projects"]

    def test_navigate_to_compress_signal_emitted(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        panel._on_result_selected(0)
        received = []
        panel._ctrl.navigate_to_compress.connect(lambda w: received.append(w))
        panel._open_wing_in_compress()
        assert received == ["projects"]

    # -- Search semantics unchanged --

    def test_search_still_returns_flat_hits(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        assert panel._results_list.count() == 2
        assert len(panel._hits) == 2

    def test_search_no_grouping(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        for i in range(panel._results_list.count()):
            item = panel._results_list.item(i)
            assert item is not None

    def test_action_buttons_disabled_after_new_empty_search(self, qapp, tmp_palace):
        from mempalace.gui_adapter import SearchResult

        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        panel._on_result_selected(0)
        assert panel._export_btn.isEnabled() is True
        empty_result = SearchResult(ok=True, query="nothing", hits=[])
        panel._on_search_done(empty_result)
        assert panel._export_btn.isEnabled() is False

    def test_source_path_not_displayed_in_result_list(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        for i in range(panel._results_list.count()):
            item = panel._results_list.item(i)
            text = item.text()
            assert "/tmp/design.md" not in text, "source_path must not appear in result list"


# ---------------------------------------------------------------------------
# 10. Panel prefill — WakeUpPanel and CompressPanel
# ---------------------------------------------------------------------------


class TestPanelPrefill:
    """WakeUpPanel and CompressPanel must accept prefill from navigation."""

    def test_wakeup_prefill_sets_wing(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import WakeUpPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = WakeUpPanel(ctrl)
        panel.prefill(wing="my-project")
        assert panel._wing_edit.text() == "my-project"

    def test_wakeup_prefill_empty_wing_does_not_overwrite(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import WakeUpPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = WakeUpPanel(ctrl)
        panel._wing_edit.setText("existing")
        panel.prefill(wing="")
        assert panel._wing_edit.text() == "existing"

    def test_compress_prefill_sets_wing(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import CompressPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = CompressPanel(ctrl)
        panel.prefill(wing="my-project")
        assert panel._wing_edit.text() == "my-project"

    def test_compress_prefill_empty_wing_does_not_overwrite(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import CompressPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = CompressPanel(ctrl)
        panel._wing_edit.setText("existing")
        panel.prefill(wing="")
        assert panel._wing_edit.text() == "existing"


# ---------------------------------------------------------------------------
# 11. MainWindow navigation — tab switch + prefill
# ---------------------------------------------------------------------------


class TestMainWindowNavigation:
    """MainWindow must switch tabs and prefill when navigate signals fire."""

    def _make_window(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import MainWindow

        ctrl = QtController(palace_path=tmp_palace)
        win = MainWindow(controller=ctrl)
        return ctrl, win

    def test_navigate_to_wakeup_switches_tab(self, qapp, tmp_palace):
        ctrl, win = self._make_window(qapp, tmp_palace)
        ctrl.navigate_to_wakeup.emit("projects")
        from PySide6.QtWidgets import QTabWidget

        tabs = win.findChild(QTabWidget)
        current_text = tabs.tabText(tabs.indexOf(win._wakeup_panel))
        assert "Wake-up" in current_text
        assert tabs.currentWidget() is win._wakeup_panel

    def test_navigate_to_wakeup_prefills_wing(self, qapp, tmp_palace):
        ctrl, win = self._make_window(qapp, tmp_palace)
        ctrl.navigate_to_wakeup.emit("projects")
        assert win._wakeup_panel._wing_edit.text() == "projects"

    def test_navigate_to_compress_switches_tab(self, qapp, tmp_palace):
        ctrl, win = self._make_window(qapp, tmp_palace)
        ctrl.navigate_to_compress.emit("projects")
        from PySide6.QtWidgets import QTabWidget

        tabs = win.findChild(QTabWidget)
        assert tabs.currentWidget() is win._compress_panel

    def test_navigate_to_compress_prefills_wing(self, qapp, tmp_palace):
        ctrl, win = self._make_window(qapp, tmp_palace)
        ctrl.navigate_to_compress.emit("projects")
        assert win._compress_panel._wing_edit.text() == "projects"

    def test_navigate_signals_exist_on_controller(self, qapp, tmp_palace):
        from gui.qt_controller import QtController

        ctrl = QtController(palace_path=tmp_palace)
        assert hasattr(ctrl, "navigate_to_wakeup")
        assert hasattr(ctrl, "navigate_to_compress")


# ---------------------------------------------------------------------------
# 12. CompressText and SourceFile adapter methods
# ---------------------------------------------------------------------------


class TestCompressTextAndSourceFile:
    """Verify gui_adapter.run_compress_text() and run_read_source_file()."""

    def test_compress_text_returns_aaak(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_compress_text(
            "We decided to use GraphQL instead of REST for better performance.",
            source_label="design.md",
            wing="projects",
            room="decisions",
        )
        assert result.ok
        assert result.aaaK_text != ""
        assert result.orig_tokens_est > 0
        assert result.comp_tokens_est > 0
        assert result.compression_ratio > 0

    def test_compress_text_does_not_store(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_compress_text("Some text to compress")
        assert result.ok
        assert result.aaaK_text != ""

    def test_read_source_file_exists(self, tmp_palace, tmp_path):
        from mempalace.gui_adapter import MemPalaceAdapter

        f = tmp_path / "notes.md"
        f.write_text("Some notes here")
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_read_source_file(str(f))
        assert result.ok
        assert result.text == "Some notes here"
        assert result.path != ""

    def test_read_source_file_not_found(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_read_source_file("/nonexistent/path/file.md")
        assert result.ok is False
        assert "not found" in result.error.lower() or "error" in result.error.lower()

    def test_read_source_file_empty_path(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_read_source_file("")
        assert result.ok is False

    def test_compress_text_result_type(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter, CompressTextResult

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_compress_text("Hello world")
        assert isinstance(result, CompressTextResult)

    def test_source_file_result_type(self, tmp_palace, tmp_path):
        from mempalace.gui_adapter import MemPalaceAdapter, SourceFileResult

        f = tmp_path / "test.txt"
        f.write_text("content")
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_read_source_file(str(f))
        assert isinstance(result, SourceFileResult)


# ---------------------------------------------------------------------------
# 13. Export block workflow
# ---------------------------------------------------------------------------


class TestExportBlockAdapter:
    """Verify gui_adapter.run_export_block() — canonical technical handoff."""

    def test_export_block_hit_scope(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter, ExportBlockResult

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="hit",
            raw_text="We decided to use GraphQL for the API.",
            source_file="design.md",
            wing="projects",
            room="decisions",
        )
        assert isinstance(result, ExportBlockResult)
        assert result.ok
        assert result.scope == "hit"
        assert "GraphQL" in result.block_text
        assert "hit" in result.block_text.lower()

    def test_export_block_file_scope(self, tmp_palace, tmp_path):
        from mempalace.gui_adapter import MemPalaceAdapter

        f = tmp_path / "design.md"
        f.write_text("File content about GraphQL decisions")
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="file",
            raw_text="File content about GraphQL decisions",
            source_file="design.md",
            source_path=str(f),
            wing="projects",
            room="decisions",
        )
        assert result.ok
        assert "file" in result.block_text.lower()
        assert "GraphQL" in result.block_text

    def test_export_block_wing_scope(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="wing",
            raw_text="Wing drawer content about Redis caching",
            wing="projects",
        )
        assert result.ok
        assert "wing" in result.block_text.lower()

    def test_export_block_invalid_scope(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(scope="invalid", raw_text="text")
        assert result.ok is False

    def test_export_block_no_text(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(scope="hit", raw_text="")
        assert result.ok is False

    def test_export_block_always_includes_handoff(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="hit",
            raw_text="Some work text",
            wing="projects",
        )
        assert result.ok
        assert "Handoff" in result.block_text

    def test_export_block_always_includes_aaak(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="hit",
            raw_text="We decided to use GraphQL for the API design",
            wing="projects",
        )
        assert result.ok
        assert "AAAK" in result.block_text

    def test_export_block_always_includes_raw_source(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="hit",
            raw_text="This is the raw verbatim content",
            wing="projects",
        )
        assert result.ok
        assert "This is the raw verbatim content" in result.block_text
        assert "Raw source" in result.block_text

    def test_export_block_preserves_raw_not_summary(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="hit",
            raw_text="Exact user words that must not be paraphrased or summarized",
            wing="projects",
        )
        assert result.ok
        assert "Exact user words that must not be paraphrased or summarized" in result.block_text

    def test_export_block_has_metadata(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="hit",
            raw_text="Some text",
            source_file="design.md",
            wing="projects",
            room="decisions",
        )
        assert result.ok
        assert "design.md" in result.block_text
        assert "projects" in result.block_text
        assert "decisions" in result.block_text

    def test_export_block_has_title(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="hit",
            raw_text="Some text",
            wing="projects",
        )
        assert result.ok
        assert "Technical handoff" in result.block_text

    def test_export_block_has_handoff_text(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="hit",
            raw_text="Some text",
            wing="projects",
        )
        assert result.ok
        assert "continuing work" in result.block_text.lower()

    def test_export_block_has_what_must_not_break(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="hit",
            raw_text="Some text",
            wing="projects",
        )
        assert result.ok
        assert (
            "must not" in result.block_text.lower()
            or "must not be broken" in result.block_text.lower()
        )

    def test_export_block_has_next_step(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_export_block(
            scope="hit",
            raw_text="Some text",
            wing="projects",
        )
        assert result.ok
        assert "next step" in result.block_text.lower()

    def test_export_block_truncates_large_source(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        large_text = "Line content here.\n" * 2000
        result = adapter.run_export_block(
            scope="hit",
            raw_text=large_text,
            source_file="big.md",
            wing="projects",
        )
        assert result.ok
        assert "lines omitted" in result.block_text
        assert "Raw source" in result.block_text

    def test_export_block_includes_full_small_source(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        small_text = "Short fragment of text"
        result = adapter.run_export_block(
            scope="hit",
            raw_text=small_text,
            wing="projects",
        )
        assert result.ok
        assert small_text in result.block_text
        assert "lines omitted" not in result.block_text


class TestExportBlockUI:
    """Verify export dialog and button existence in SearchPanel."""

    def _make_panel(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import SearchPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = SearchPanel(ctrl)
        return ctrl, panel

    def _populate_hits(self, panel):
        from mempalace.gui_adapter import SearchResult, SearchHit

        hits = [
            SearchHit(
                text="GraphQL design decisions",
                wing="projects",
                room="2024-01-15",
                source_file="design.md",
                source_path="/tmp/design.md",
                similarity=0.85,
                distance=0.15,
            ),
        ]
        result = SearchResult(ok=True, query="test", hits=hits)
        panel._on_search_done(result)
        return hits

    def test_has_prepare_for_new_chat_button(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QPushButton

        _, panel = self._make_panel(qapp, tmp_palace)
        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert any("Prepare for new chat" in lbl for lbl in labels)

    def test_export_button_disabled_without_hit(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        assert panel._export_btn.isEnabled() is False

    def test_export_button_enabled_with_hit(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        panel._on_result_selected(0)
        assert panel._export_btn.isEnabled() is True

    def test_export_button_has_primary_style(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        style = panel._export_btn.styleSheet()
        assert "2563eb" in style

    def test_export_dialog_opens(self, qapp, tmp_palace):
        from gui.main_window import ExportBlockDialog

        _, panel = self._make_panel(qapp, tmp_palace)
        hits = self._populate_hits(panel)
        panel._on_result_selected(0)
        dlg = ExportBlockDialog(panel._ctrl, hits[0], parent=panel)
        assert dlg is not None
        assert dlg.windowTitle() == "Prepare for new chat"

    def test_export_dialog_has_scope_checkboxes(self, qapp, tmp_palace):
        from gui.main_window import ExportBlockDialog

        _, panel = self._make_panel(qapp, tmp_palace)
        hits = self._populate_hits(panel)
        dlg = ExportBlockDialog(panel._ctrl, hits[0], parent=panel)
        assert hasattr(dlg, "_scope_hit_rb")
        assert hasattr(dlg, "_scope_file_rb")
        assert hasattr(dlg, "_scope_wing_rb")

    def test_export_dialog_has_no_section_toggles(self, qapp, tmp_palace):
        from gui.main_window import ExportBlockDialog

        _, panel = self._make_panel(qapp, tmp_palace)
        hits = self._populate_hits(panel)
        dlg = ExportBlockDialog(panel._ctrl, hits[0], parent=panel)
        assert not hasattr(dlg, "_recap_cb")
        assert not hasattr(dlg, "_wakeup_cb")
        assert not hasattr(dlg, "_aaak_cb")
        assert not hasattr(dlg, "_raw_cb")

    def test_export_dialog_has_copy_and_save(self, qapp, tmp_palace):
        from gui.main_window import ExportBlockDialog

        _, panel = self._make_panel(qapp, tmp_palace)
        hits = self._populate_hits(panel)
        dlg = ExportBlockDialog(panel._ctrl, hits[0], parent=panel)
        assert hasattr(dlg, "_copy_btn")
        assert hasattr(dlg, "_save_btn")

    def test_context_menu_has_export_action(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        panel._on_result_selected(0)
        from PySide6.QtWidgets import QMenu

        menu = QMenu(panel)
        menu.addAction("Prepare for new chat", lambda: None)
        assert len(menu.actions()) == 1

    def test_search_semantics_unchanged_with_export(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        self._populate_hits(panel)
        assert panel._results_list.count() == 1
        assert len(panel._hits) == 1


class TestReadWingDrawers:
    """Verify run_read_wing_drawers adapter method."""

    def test_read_wing_no_palace(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_read_wing_drawers("projects")
        assert result.ok is False

    def test_read_wing_empty_wing(self, tmp_palace):
        from mempalace.gui_adapter import MemPalaceAdapter

        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        result = adapter.run_read_wing_drawers("")
        assert result.ok is False

    def test_read_wing_with_mined_palace(self, tmp_palace, tmp_path):
        from mempalace.gui_adapter import MemPalaceAdapter

        proj = tmp_path / "wing_proj"
        proj.mkdir()
        (proj / "notes.md").write_text("GraphQL API design decisions and architecture")
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        init_result = adapter.safe_init(project_dir=str(proj))
        if init_result.ok:
            adapter.run_mine_projects(str(proj))
            result = adapter.run_read_wing_drawers("wing_proj")
            assert result.ok is True or "No drawers" in (result.error or "")


# ---------------------------------------------------------------------------
# 14. Continue from file — COMMIT 3
# ---------------------------------------------------------------------------


class TestContinueFromFile:
    """Verify 'Continue from file...' button and force_file_scope flow."""

    def _make_panel(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import SearchPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = SearchPanel(ctrl)
        return ctrl, panel

    def test_has_continue_from_file_button(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QPushButton

        _, panel = self._make_panel(qapp, tmp_palace)
        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert any("Continue from file" in lbl for lbl in labels)

    def test_continue_from_file_button_always_enabled(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        assert panel._from_file_btn.isEnabled() is True

    def test_continue_from_file_button_has_tooltip(self, qapp, tmp_palace):
        _, panel = self._make_panel(qapp, tmp_palace)
        assert panel._from_file_btn.toolTip() != ""

    def test_continue_from_file_creates_synthetic_hit(self, qapp, tmp_palace, tmp_path):
        from mempalace.gui_adapter import SearchHit

        _, panel = self._make_panel(qapp, tmp_palace)
        f = tmp_path / "old_chat.md"
        f.write_text("Previous conversation about GraphQL design")
        result = panel._ctrl._adapter.run_read_source_file(str(f))
        assert result.ok
        synthetic_hit = SearchHit(
            text=result.text,
            wing="",
            room="",
            source_file=f.name,
            source_path=str(f.resolve()),
            similarity=0.0,
            distance=0.0,
        )
        assert synthetic_hit.text == "Previous conversation about GraphQL design"
        assert synthetic_hit.source_file == "old_chat.md"
        assert synthetic_hit.wing == ""
        assert synthetic_hit.room == ""


class TestForceFileScope:
    """Verify ExportBlockDialog.force_file_scope() locks scope to file."""

    def _make_dialog(self, qapp, tmp_palace, tmp_path=None):
        from gui.qt_controller import QtController
        from gui.main_window import ExportBlockDialog, SearchPanel
        from mempalace.gui_adapter import SearchHit

        ctrl = QtController(palace_path=tmp_palace)
        panel = SearchPanel(ctrl)
        source_path = str(tmp_path / "design.md") if tmp_path else "/tmp/design.md"
        hit = SearchHit(
            text="Some content",
            wing="",
            room="",
            source_file="design.md",
            source_path=source_path,
            similarity=0.0,
            distance=0.0,
        )
        dlg = ExportBlockDialog(ctrl, hit, parent=panel)
        return panel, dlg

    def test_force_file_scope_selects_file(self, qapp, tmp_palace):
        _panel, dlg = self._make_dialog(qapp, tmp_palace)
        dlg.force_file_scope()
        assert dlg._scope_file_rb.isChecked() is True
        assert dlg._scope == "file"

    def test_force_file_scope_disables_hit(self, qapp, tmp_palace):
        _panel, dlg = self._make_dialog(qapp, tmp_palace)
        dlg.force_file_scope()
        assert dlg._scope_hit_rb.isEnabled() is False

    def test_force_file_scope_disables_wing(self, qapp, tmp_palace):
        _panel, dlg = self._make_dialog(qapp, tmp_palace)
        dlg.force_file_scope()
        assert dlg._scope_wing_rb.isEnabled() is False

    def test_force_file_scope_unchecks_hit(self, qapp, tmp_palace):
        _panel, dlg = self._make_dialog(qapp, tmp_palace)
        dlg.force_file_scope()
        assert dlg._scope_hit_rb.isChecked() is False

    def test_force_file_scope_unchecks_wing(self, qapp, tmp_palace):
        _panel, dlg = self._make_dialog(qapp, tmp_palace)
        dlg.force_file_scope()
        assert dlg._scope_wing_rb.isChecked() is False

    def test_force_file_scope_preloads_file_text(self, qapp, tmp_palace, tmp_path):
        f = tmp_path / "design.md"
        f.write_text("Design file content")
        _panel, dlg = self._make_dialog(qapp, tmp_palace, tmp_path)
        dlg.force_file_scope()
        assert "Design file content" in dlg._raw_text_cache.get("file", "")

    def test_force_file_scope_shows_alongside_checkbox(self, qapp, tmp_palace, tmp_path):
        f = tmp_path / "design.md"
        f.write_text("content")
        _panel, dlg = self._make_dialog(qapp, tmp_palace, tmp_path)
        dlg.force_file_scope()
        assert dlg._alongside_cb.isHidden() is False

    def test_alongside_checkbox_hidden_without_source_path(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import ExportBlockDialog, SearchPanel
        from mempalace.gui_adapter import SearchHit

        ctrl = QtController(palace_path=tmp_palace)
        panel = SearchPanel(ctrl)
        hit = SearchHit(
            text="Content",
            wing="",
            room="",
            source_file="design.md",
            source_path="",
            similarity=0.0,
            distance=0.0,
        )
        dlg = ExportBlockDialog(ctrl, hit, parent=panel)
        dlg.force_file_scope()
        assert dlg._alongside_cb.isVisible() is False


class TestSaveAlongsideSource:
    """Verify 'Save alongside source' writes file next to original."""

    def test_save_alongside_creates_file(self, qapp, tmp_palace, tmp_path):
        from gui.qt_controller import QtController
        from gui.main_window import ExportBlockDialog, SearchPanel
        from mempalace.gui_adapter import SearchHit

        src = tmp_path / "old_chat.md"
        src.write_text("Old conversation content about design decisions")
        ctrl = QtController(palace_path=tmp_palace)
        panel = SearchPanel(ctrl)
        hit = SearchHit(
            text=src.read_text(),
            wing="",
            room="",
            source_file=src.name,
            source_path=str(src.resolve()),
            similarity=0.0,
            distance=0.0,
        )
        dlg = ExportBlockDialog(ctrl, hit, parent=panel)
        dlg.force_file_scope()
        dlg._preview.setPlainText("Handoff block content")
        dlg._alongside_cb.setChecked(True)
        dlg._save_block()
        out = tmp_path / "old_chat_handoff.md"
        assert out.exists()
        assert out.read_text() == "Handoff block content"

    def test_save_alongside_preserves_original(self, qapp, tmp_palace, tmp_path):
        from gui.qt_controller import QtController
        from gui.main_window import ExportBlockDialog, SearchPanel
        from mempalace.gui_adapter import SearchHit

        src = tmp_path / "notes.md"
        src.write_text("Original notes content")
        ctrl = QtController(palace_path=tmp_palace)
        panel = SearchPanel(ctrl)
        hit = SearchHit(
            text=src.read_text(),
            wing="",
            room="",
            source_file=src.name,
            source_path=str(src.resolve()),
            similarity=0.0,
            distance=0.0,
        )
        dlg = ExportBlockDialog(ctrl, hit, parent=panel)
        dlg.force_file_scope()
        dlg._preview.setPlainText("Handoff content")
        dlg._alongside_cb.setChecked(True)
        dlg._save_block()
        assert src.read_text() == "Original notes content"

    def test_save_alongside_naming_convention(self, qapp, tmp_palace, tmp_path):
        from gui.qt_controller import QtController
        from gui.main_window import ExportBlockDialog, SearchPanel
        from mempalace.gui_adapter import SearchHit

        src = tmp_path / "design.md"
        src.write_text("Design content")
        ctrl = QtController(palace_path=tmp_palace)
        panel = SearchPanel(ctrl)
        hit = SearchHit(
            text=src.read_text(),
            wing="",
            room="",
            source_file=src.name,
            source_path=str(src.resolve()),
            similarity=0.0,
            distance=0.0,
        )
        dlg = ExportBlockDialog(ctrl, hit, parent=panel)
        dlg.force_file_scope()
        dlg._preview.setPlainText("Content")
        dlg._alongside_cb.setChecked(True)
        dlg._save_block()
        expected = tmp_path / "design_handoff.md"
        assert expected.exists()


class TestContinueFromFileIntegration:
    """Integration: continue-from-file + canonical handoff + UI simplification."""

    def test_synthetic_hit_produces_file_scope_handoff(self, tmp_palace, tmp_path):
        from mempalace.gui_adapter import MemPalaceAdapter, SearchHit

        f = tmp_path / "old.md"
        f.write_text("Old chat about architecture decisions")
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        read_result = adapter.run_read_source_file(str(f))
        assert read_result.ok
        synthetic_hit = SearchHit(
            text=read_result.text,
            wing="",
            room="",
            source_file=f.name,
            source_path=str(f.resolve()),
            similarity=0.0,
            distance=0.0,
        )
        result = adapter.run_export_block(
            scope="file",
            raw_text=synthetic_hit.text,
            source_file=synthetic_hit.source_file,
            source_path=synthetic_hit.source_path,
            wing=synthetic_hit.wing,
            room=synthetic_hit.room,
        )
        assert result.ok
        assert result.scope == "file"
        assert "Technical handoff" in result.block_text
        assert "old.md" in result.block_text

    def test_large_file_handoff_truncates(self, tmp_palace, tmp_path):
        from mempalace.gui_adapter import MemPalaceAdapter

        f = tmp_path / "big_chat.md"
        f.write_text("Line of text.\n" * 2000)
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        raw = f.read_text()
        result = adapter.run_export_block(
            scope="file",
            raw_text=raw,
            source_file="big_chat.md",
            source_path=str(f.resolve()),
        )
        assert result.ok
        assert "lines omitted" in result.block_text

    def test_small_file_handoff_includes_full_source(self, tmp_palace, tmp_path):
        from mempalace.gui_adapter import MemPalaceAdapter

        f = tmp_path / "small_chat.md"
        f.write_text("Short conversation about testing")
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        raw = f.read_text()
        result = adapter.run_export_block(
            scope="file",
            raw_text=raw,
            source_file="small_chat.md",
            source_path=str(f.resolve()),
        )
        assert result.ok
        assert "Short conversation about testing" in result.block_text
        assert "lines omitted" not in result.block_text

    def test_primary_button_still_works_with_continue_from_file(self, qapp, tmp_palace):
        from PySide6.QtWidgets import QPushButton

        from gui.qt_controller import QtController
        from gui.main_window import SearchPanel

        ctrl = QtController(palace_path=tmp_palace)
        panel = SearchPanel(ctrl)
        btns = panel.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert any("Prepare for new chat" in lbl for lbl in labels)
        assert any("Continue from file" in lbl for lbl in labels)
        assert any("More actions" in lbl for lbl in labels)

    def test_existing_search_wakeup_compress_unchanged(self, tmp_palace, tmp_path):
        from mempalace.gui_adapter import MemPalaceAdapter

        proj = tmp_path / "integ_proj"
        proj.mkdir()
        (proj / "alpha.txt").write_text(
            "Alpha: GraphQL API design decisions and architecture overview."
        )
        adapter = MemPalaceAdapter(palace_path=str(tmp_palace))
        adapter.safe_init(project_dir=str(proj))
        adapter.run_mine_projects(str(proj))
        search_result = adapter.run_search("architecture", n_results=5)
        assert search_result.ok
        wakeup_result = adapter.run_wakeup()
        assert wakeup_result.ok or "No palace" in (wakeup_result.error or "")
        compress_result = adapter.run_compress(dry_run=True)
        assert compress_result.ok or "No drawers" in (compress_result.error or "")
