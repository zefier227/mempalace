"""
gui/app.py -- MemPalace GUI v1 entry point.
============================================

Usage
-----
Run directly::

    python -m gui.app                        # uses default palace path
    python -m gui.app --palace ~/my_palace   # explicit palace path
    python -m gui.app --palace ~/my_palace --debug

Or via the installed script (if added to pyproject.toml)::

    mempalace-gui

Architecture
------------
One ``QApplication``, one ``QtController`` (owns the ``MemPalaceAdapter``),
one ``MainWindow``.  All blocking I/O runs on ``QThread`` workers inside
``QtController``; the main thread only drives the Qt event loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _configure_logging(debug: bool) -> None:
    """Set up root logger before creating QApplication."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mempalace-gui",
        description="MemPalace GUI v1 -- init, mine, status, search.",
    )
    parser.add_argument(
        "--palace",
        metavar="PATH",
        default=None,
        help=(
            "Path to the palace directory.  "
            "Defaults to the value in mempalace.yaml or ~/.mempalace/palace."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable verbose debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Application entry point.  Returns the QApplication exit code.

    Separated from ``if __name__ == '__main__'`` so it can be called
    programmatically in tests (with a ``QApplication`` already running).
    """
    args = _parse_args(argv)
    _configure_logging(args.debug)

    log = logging.getLogger("mempalace.gui.app")
    log.debug("Starting MemPalace GUI, palace=%s", args.palace)

    # Resolve palace path early so the controller and window title are right.
    palace_path: str | None = None
    if args.palace:
        palace_path = str(Path(args.palace).expanduser().resolve())
        log.debug("Explicit palace path: %s", palace_path)

    # ------------------------------------------------------------------
    # Qt setup -- import *after* logging so early import errors are visible.
    # ------------------------------------------------------------------
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError as exc:  # pragma: no cover
        print(
            f"ERROR: PySide6 is not installed.  "
            f"Run: pip install 'PySide6>=6.7'\n{exc}",
            file=sys.stderr,
        )
        return 2

    # Allow running headless (e.g. CI / offscreen tests)
    if "QT_QPA_PLATFORM" not in os.environ and sys.platform != "darwin":
        # On non-macOS without explicit platform, default to offscreen to
        # avoid display-not-found crashes in CI.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("MemPalace")
    app.setOrganizationName("MemPalace")

    # High-DPI is on by default in Qt6; nothing extra needed.

    # ------------------------------------------------------------------
    # Controller + window
    # ------------------------------------------------------------------
    from gui.qt_controller import QtController
    from gui.main_window import MainWindow

    ctrl = QtController(palace_path=palace_path)
    window = MainWindow(controller=ctrl)
    window.show()

    log.debug("Entering Qt event loop")
    exit_code = app.exec()
    log.debug("Qt event loop exited with code %d", exit_code)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
