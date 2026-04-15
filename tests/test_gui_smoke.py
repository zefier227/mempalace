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
