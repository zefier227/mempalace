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
import sys
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
        # adapter returns dict with at least one of these keys
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
        assert tabs.count() == 4

    def test_tab_labels(self, qapp, tmp_palace):
        from gui.qt_controller import QtController
        from gui.main_window import MainWindow
        from PySide6.QtWidgets import QTabWidget
        ctrl = QtController(palace_path=tmp_palace)
        win = MainWindow(controller=ctrl)
        tabs = win.findChild(QTabWidget)
        labels = [tabs.tabText(i) for i in range(tabs.count())]
        assert any("Init" in l for l in labels)
        assert any("Mine" in l for l in labels)
        assert any("Status" in l for l in labels)
        assert any("Search" in l for l in labels)

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
        # At least query field + wing filter
        assert len(fields) >= 2

    def test_search_panel_has_n_results_spinbox(self, qapp, tmp_palace):
        from gui.main_window import SearchPanel
        from PySide6.QtWidgets import QSpinBox
        panel = SearchPanel(self._make_ctrl(qapp, tmp_palace))
        spinboxes = panel.findChildren(QSpinBox)
        assert len(spinboxes) >= 1
        spin = spinboxes[0]
        assert spin.value() == 50
        assert spin.minimum() >= 1

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
            "Post-mine status refresh must use QTimer.singleShot "
            "to avoid busy-flag race"
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
# 6. Search selection / preview behavior (SearchPanel)
# ---------------------------------------------------------------------------


class TestSearchSelectionBehavior:
    """Verify that clicking a result switches the preview to that group."""

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

        result = adapter.run_search("architecture", n_results=50)
        return result

    def test_search_result_groups_populated(self, qapp, tmp_palace, tmp_path):
        from mempalace.gui_adapter import SearchResult
        from gui.main_window import SearchPanel
        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        assert len(panel._groups) >= 1

    def test_click_group_shows_correct_preview(self, qapp, tmp_palace, tmp_path):
        from mempalace.gui_adapter import SearchResult
        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        if len(panel._groups) < 2:
            pytest.skip("Need at least 2 groups for selection test")
        panel._on_result_selected(0)
        first_preview = panel._preview.toPlainText()
        panel._on_result_selected(1)
        second_preview = panel._preview.toPlainText()
        assert first_preview != second_preview or len(panel._groups) == 1

    def test_switching_group_resets_chunk_index(self, qapp, tmp_palace, tmp_path):
        from mempalace.gui_adapter import SearchResult
        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        panel._on_result_selected(0)
        # Even if chunk_index was advanced on group 0, switching to group 1 resets
        panel._show_group(0, chunk_index=0)
        if len(panel._groups[0].all_hits) > 1:
            panel._show_group(0, chunk_index=1)
        if len(panel._groups) > 1:
            panel._on_result_selected(1)
            assert getattr(panel._groups[1], "_active_chunk", 0) == 0

    def test_new_query_clears_old_preview(self, qapp, tmp_palace, tmp_path):
        from mempalace.gui_adapter import SearchResult
        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        assert len(panel._groups) >= 1
        panel._on_result_selected(0)
        assert panel._preview.toPlainText() != ""

        new_result = SearchResult(ok=True, query="new", hits=[], groups=[])
        panel._on_search_done(new_result)
        assert panel._preview.toPlainText() == ""
        assert panel._preview_header.text() == ""

    def test_preview_header_shows_file_name(self, qapp, tmp_palace, tmp_path):
        from mempalace.gui_adapter import SearchResult
        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        if not panel._groups:
            pytest.skip("No groups in result")
        panel._on_result_selected(0)
        header_text = panel._preview_header.text()
        assert panel._groups[0].source_file in header_text

    def test_list_items_contain_snippet(self, qapp, tmp_palace, tmp_path):
        from mempalace.gui_adapter import SearchResult
        ctrl, panel = self._make_panels(qapp, tmp_palace)
        result = self._populate_and_search(ctrl, panel, tmp_palace, tmp_path)
        panel._on_search_done(result)
        for i in range(panel._results_list.count()):
            text = panel._results_list.item(i).text()
            assert '"' in text  # snippet is quoted in list items


# ---------------------------------------------------------------------------
# 7. Informative snippet logic
# ---------------------------------------------------------------------------


class TestInformativeSnippet:
    """_informative_snippet should pick content-rich lines."""

    def test_picks_longer_content_line(self):
        from mempalace.gui_adapter import _informative_snippet
        text = "# Header\nA substantial line with real content about GraphQL\n\n"
        result = _informative_snippet(text)
        assert "substantial" in result or "GraphQL" in result

    def test_short_lines_only(self):
        from mempalace.gui_adapter import _informative_snippet
        text = "short line\nanother brief line\n"
        result = _informative_snippet(text, max_len=40)
        assert len(result) > 0

    def test_empty_text(self):
        from mempalace.gui_adapter import _informative_snippet
        assert _informative_snippet("") == ""

    def test_truncates_long_line(self):
        from mempalace.gui_adapter import _informative_snippet
        text = "A" * 200
        result = _informative_snippet(text, max_len=40)
        assert len(result) <= 40
        assert result.endswith("...")
