"""
py2app build configuration for MemPalace macOS .app bundle.

This file is ONLY used by ``python setup.py py2app`` to produce the
MemPalace.app bundle.  It does NOT replace or conflict with the
hatchling-based build in pyproject.toml.

Strategy
--------
* ``packages`` — copies the entire package.  Only list packages whose
  ``__file__`` attribute is a real path (not a namespace package), because
  py2app uses ``imp.find_module`` which cannot handle namespace packages.
  Namespace packages (google, opentelemetry) are copied in a post-build step.
* ``includes`` — specific submodules from namespace packages and modules
  loaded dynamically by native extensions.
* ``excludes`` — strip test frameworks and unused stdlib modules.

Post-build fixes
----------------
1. ``install_name_tool`` to fix the ``python`` binary's dylib reference
   (py2app bug: it points to ``@executable_path/../../../../Python3``
   instead of the correct Frameworks/ path).
2. Copy namespace packages (google, opentelemetry) into the bundle.
3. Ad-hoc code-sign the bundle.

Build commands
--------------
    python setup.py py2app          # standalone .app (includes post-build fixes)
    python setup.py py2app -A       # alias (symlink) mode for dev
"""

import os
import shutil
import subprocess
import sys

sys.setrecursionlimit(6000)

from setuptools import setup
from py2app.build_app import py2app as _py2app


NAMESPACE_PACKAGES = ["google", "opentelemetry"]


class py2app(_py2app):
    def run(self):
        _py2app.run(self)
        self._post_build_fixes()

    def _post_build_fixes(self):
        dist_dir = os.path.join(os.getcwd(), "dist", "MemPalace.app")
        if not os.path.isdir(dist_dir):
            return
        contents = os.path.join(dist_dir, "Contents")
        macos_dir = os.path.join(contents, "MacOS")
        frameworks_python = os.path.join(
            contents, "Frameworks", "Python3.framework",
            "Versions", "3.9", "Python3",
        )
        python_bin = os.path.join(macos_dir, "python")
        site_packages = os.path.join(
            contents, "Resources", "lib", "python3.9",
        )
        user_site = os.path.expanduser(
            "~/Library/Python/3.9/lib/python/site-packages"
        )

        # Fix 1: install_name_tool on the python binary
        if os.path.isfile(python_bin) and os.path.isfile(frameworks_python):
            old_ref = "@executable_path/../../../../Python3"
            new_ref = (
                "@executable_path/../Frameworks/"
                "Python3.framework/Versions/3.9/Python3"
            )
            try:
                subprocess.check_call([
                    "install_name_tool", "-change",
                    old_ref, new_ref, python_bin,
                ])
                print("  [post-build] Fixed python binary dylib reference")
            except subprocess.CalledProcessError as e:
                print(f"  [post-build] WARNING: install_name_tool failed: {e}")

        # Fix 2: copy namespace packages into the bundle
        for pkg in NAMESPACE_PACKAGES:
            src = os.path.join(user_site, pkg)
            dst = os.path.join(site_packages, pkg)
            if os.path.isdir(src) and not os.path.isdir(dst):
                shutil.copytree(src, dst, symlinks=False)
                print(f"  [post-build] Copied namespace package: {pkg}")

        # Fix 2b: remove namespace package .pyc entries from python39.zip
        # so the full filesystem copies take precedence
        zip_path = os.path.join(
            contents, "Resources", "lib", "python39.zip"
        )
        if os.path.isfile(zip_path):
            import zipfile
            tmp_path = zip_path + ".tmp"
            try:
                with zipfile.ZipFile(zip_path, 'r') as zin:
                    keep = [
                        n for n in zin.namelist()
                        if not any(n.startswith(pkg + "/") for pkg in NAMESPACE_PACKAGES)
                    ]
                    with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zout:
                        for item in keep:
                            zout.writestr(item, zin.read(item))
                os.replace(tmp_path, zip_path)
                print(f"  [post-build] Cleaned {len(NAMESPACE_PACKAGES)} namespace entries from zip")
            except Exception as e:
                print(f"  [post-build] WARNING: zip cleanup failed: {e}")

        # Fix 3: ad-hoc code-sign
        try:
            subprocess.check_call([
                "codesign", "--force", "--deep", "--sign", "-", dist_dir,
            ])
            print("  [post-build] Ad-hoc code-signed the bundle")
        except subprocess.CalledProcessError as e:
            print(f"  [post-build] WARNING: codesign failed: {e}")


APP = ["gui/app.py"]

OPTIONS = {
    "packages": [
        "mempalace",
        "gui",
        "chromadb",
        "chromadb_rust_bindings",
        "PySide6",
        "shiboken6",
        "onnxruntime",
        "numpy",
        "grpc",
        "yaml",
        "pydantic",
        "pydantic_core",
        "pydantic_settings",
        "httpx",
        "httpcore",
        "h11",
        "anyio",
        "tokenizers",
        "tqdm",
        "typer",
        "click",
        "rich",
        "bcrypt",
        "mmh3",
        "orjson",
        "overrides",
        "pypika",
        "tenacity",
        "jsonschema",
        "importlib_resources",
        "pybase64",
        "uvicorn",
        "starlette",
        "certifi",
        "idna",
        "kubernetes",
        "annotated_types",
        "typing_extensions",
    ],
    "includes": [
        "chromadb_rust_bindings.abi3",
    ],
    "excludes": [
        "tkinter",
        "test",
        "tests",
        "unittest",
        "pytest",
        "idlelib",
        "distutils",
        "setuptools",
        "pip",
        "wheel",
        "_pytest",
    ],
    "plist": {
        "CFBundleName": "MemPalace",
        "CFBundleShortVersionString": "3.3.0",
        "CFBundleIdentifier": "com.mempalace.app",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundlePackageType": "APPL",
        "CFBundleSignature": "MPlc",
        "LSMinimumSystemVersion": "10.15.0",
        "NSHighResolutionCapable": True,
        "NSSupportsAutomaticGraphicsSwitching": True,
        "LSUIElement": False,
    },
}

setup(
    name="MemPalace",
    app=APP,
    cmdclass={"py2app": py2app},
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
