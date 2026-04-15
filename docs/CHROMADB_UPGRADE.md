# ChromaDB Upgrade: 0.6.x → 1.5.x

**Status**: Complete — `pyproject.toml` now requires `chromadb>=1.5.4`.  
**Date**: 2026-04-15  
**Affects**: macOS ARM64 (.app packaging), backend stability, migration path.

---

## 1. Problem Summary

MemPalace originally pinned `chromadb>=0.5.0` and `uv.lock` resolved to
`chromadb==0.6.3` (with `chroma-hnswlib==0.7.6`). On macOS ARM64 this version
reliably crashes with a null-pointer segfault inside the Rust shared library
`chromadb_rust_bindings.abi3.so` (triggered by the hnswlib `updatePoint` code
path). There was no workaround: the env-var `ORT_DISABLE_COREML=1` does not
suppress this crash, and the ONNX Runtime CoreML EP is a red herring — the
real fault is in hnswlib's ARM64 threading code.

Separately, ChromaDB 0.6.x stores `seq_id` fields as big-endian 8-byte BLOBs.
ChromaDB 1.5.x expects INTEGER. When a 0.6.x database is opened with 1.5.x,
the Rust compactor crashes immediately because the type mismatch is caught at
init time.

---

## 2. Fix Applied

### 2a. Dependency floor raised

`pyproject.toml`:

```toml
dependencies = [
    # chromadb>=1.5.4 fixes macOS ARM64 segfaults (hnswlib -> Rust bindings).
    "chromadb>=1.5.4",
    "pyyaml>=6.0,<7",
]
```

ChromaDB 1.5.x replaces `chroma-hnswlib` with a pure-Rust HNSW implementation.
The segfault-prone hnswlib `.so` is no longer shipped. Both x86_64 and ARM64
are served by a single `abi3` wheel — no platform-specific binary quirks.

### 2b. BLOB seq_id migration guard

`mempalace/backends/chroma.py` — `_fix_blob_seq_ids(palace_path)`:

```python
def _fix_blob_seq_ids(palace_path: str):
    """Fix ChromaDB 0.6.x → 1.5.x migration bug: BLOB seq_ids → INTEGER."""
    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return
    with sqlite3.connect(db_path) as conn:
        for table in ("embeddings", "max_seq_id"):
            rows = conn.execute(
                f"SELECT rowid, seq_id FROM {table} WHERE typeof(seq_id) = 'blob'"
            ).fetchall()
            if not rows:
                continue
            updates = [
                (int.from_bytes(blob, byteorder="big"), rowid)
                for rowid, blob in rows
            ]
            conn.executemany(f"UPDATE {table} SET seq_id = ? WHERE rowid = ?", updates)
        conn.commit()
```

This function runs **before** `PersistentClient` is created (in
`ChromaBackend._client()` and `ChromaBackend.make_client()`). On a
fresh 1.5.x database the `WHERE typeof(seq_id) = 'blob'` filter returns zero
rows, so the function is a near-zero-cost no-op. On a migrated 0.6.x database
it converts the old BLOB values to integers and commits before the compactor
fires.

---

## 3. Files Changed

| File | Change |
|------|--------|
| `pyproject.toml` | `chromadb>=0.5.0` → `chromadb>=1.5.4`; added `pytest-timeout>=2.1` to dev deps; added `--timeout=120` to pytest addopts |
| `mempalace/backends/chroma.py` | Added `_fix_blob_seq_ids()`; rewrote `ChromaBackend` to cache clients per palace path and call the fix before each new `PersistentClient` |
| `mempalace/gui_adapter.py` | New file: thin adapter layer for macOS GUI (see §5) |
| `tests/test_gui_adapter.py` | New file: 41 smoke tests covering init, mine, search, status, MCP server |
| `docs/CHROMADB_UPGRADE.md` | This document |

---

## 4. Risk Analysis

### Low risk: upgrading in a fresh environment

If the user has never used MemPalace before (no `~/.mempalace/palace/`), the
upgrade is fully transparent. ChromaDB 1.5.x creates a new SQLite database with
INTEGER `seq_id` columns from the start.

### Medium risk: existing 0.6.x palace

Users with an existing palace created under 0.6.x will have BLOB `seq_id`
values. The `_fix_blob_seq_ids()` guard converts them automatically before
opening the client. The conversion is:

```
new_integer = int.from_bytes(old_blob, byteorder="big")
```

This is the same encoding ChromaDB 0.6.x used internally, so no data is lost.

**Manual rollback** (if migration fails):

1. Stop all MemPalace processes.
2. `cp -r ~/.mempalace/palace ~/.mempalace/palace.bak`  ← backup first!
3. Pin old version: `pip install 'chromadb==0.6.3'`
4. Remove `chroma-hnswlib` conflicts if needed.
5. Open `~/.mempalace/palace/chroma.sqlite3` with `sqlite3` and verify
   `typeof(seq_id)` in `embeddings` — should be `'blob'` on the old client.

**Testing before production**:

```bash
# 1. Backup palace
cp -r ~/.mempalace/palace ~/.mempalace/palace.bak

# 2. Install new version
pip install 'chromadb>=1.5.4'

# 3. Run the smoke test (uses a temp palace, safe)
python -m pytest tests/test_gui_adapter.py -v

# 4. Test against your real palace
python -c "
from mempalace.gui_adapter import MemPalaceAdapter
a = MemPalaceAdapter()
s = a.run_status()
print('ok:', s.ok, 'drawers:', s.total_drawers, 'chromadb:', s.chromadb_version)
"
```

### API changes: chromadb 0.6.x → 1.5.x

| Feature | 0.6.x | 1.5.x | Impact |
|---------|-------|-------|--------|
| `hnsw:space` metadata | accepted | accepted | ✅ no change |
| `get_collection()` | raises `ValueError` if missing | raises `chromadb.errors.NotFoundError` | Handled by adapter's `try/except Exception` |
| `PersistentClient` | creates on-disk HNSW index | uses Rust compactor | ✅ transparent |
| Thread safety | per-process file lock | per-process file lock (stricter) | See §6 |
| `hnswlib` package | required | **not needed** | ✅ removed from deps |
| Telemetry | Posthog (chatty) | Posthog (silenced) | `__init__.py` sets logger to CRITICAL |

---

## 5. Adapter Layer (`mempalace/gui_adapter.py`)

The adapter provides a clean, GUI-safe interface over the MemPalace backend:

### Action → Entry Point → I/O → Mode

| Action | Entry Point | Input | Output | Errors | Mode |
|--------|-------------|-------|--------|--------|------|
| `safe_init` | `MempalaceConfig().init()` + `detect_rooms_local(yes=True)` | `palace_path`, optional `project_dir` | `InitResult(ok, config_path, palace_path)` | `InitResult(ok=False, error=…)` | Direct Python, non-blocking |
| `run_mine_projects` | `python -m mempalace mine --mode projects` | `project_dir`, wing, limit, dry_run | `MineResult(ok, files_processed, drawers_filed, wing, rooms)` + `MineProgressEvent` per file | `MineResult(ok=False, error=…)` | **Subprocess**, streaming stdout |
| `run_mine_convos` | `python -m mempalace mine --mode convos` | `convo_dir`, wing, extract_mode | `MineResult` + `MineProgressEvent` | `MineResult(ok=False, error=…)` | **Subprocess**, streaming stdout |
| `run_search` | `searcher.search_memories()` | query, wing, room, n_results | `SearchResult(ok, query, hits, total_candidates)` | `SearchResult(ok=False, error=…)` | **Direct Python**, non-blocking |
| `run_status` | `palace.get_collection()` + metadata scan | — | `PalaceStatus(ok, total_drawers, wings, chromadb_version)` | `PalaceStatus(ok=False, error=…)` | **Direct Python**, non-blocking |
| `start_mcp_server` | `python -m mempalace.mcp_server` | `palace_path`, extra_args | `McpServerStatus(running, pid)` | `McpServerStatus(running=False, error=…)` | **Subprocess** (long-lived) |
| `stop_mcp_server` | `SIGTERM` → `SIGKILL` | timeout | `McpServerStatus(running=False)` | `McpServerStatus(running=True, error=…)` | Signal to subprocess |

### Key design decisions

- **`run_search` is direct Python** (not subprocess) because `search_memories()`
  is a pure read operation with no side effects, takes <100 ms, and returns a
  dict. The adapter wraps it in `SearchResult`.
- **`run_status` is direct Python** because it reads ChromaDB metadata with
  `col.get(include=["metadatas"])`, which is fast and read-only.
- **`run_mine_*` is subprocess** because mine is long-running (seconds to
  minutes), prints progress lines to stdout, and must be cancellable via
  `proc.kill()` from the GUI without blocking the event loop.
- **`mcp_server` is subprocess** because it is a persistent JSON-RPC daemon
  that reads from stdin/writes to stdout. It must outlive any individual GUI
  operation and be stoppable on demand.
- **`safe_init` is direct Python** because `MempalaceConfig().init()` only
  creates directories and writes JSON — no interactive prompts. Room detection
  is called with `yes=True` to suppress all `input()` calls.

---

## 6. Thread Safety Notes (ChromaDB 1.5.x)

ChromaDB 1.5.x uses a Rust backend with **process-level file locking** on
`chroma.sqlite3`. Two `PersistentClient` instances in the same process pointing
to the same palace path will fail with `Could not connect to tenant`.

**Rules for the GUI**:

1. **One `MemPalaceAdapter` instance per palace** — the adapter caches the
   `ChromaBackend` client, so multiple calls reuse the same connection.
2. **Search and status from the main thread** (or a single worker thread) —
   do not issue concurrent `run_search` calls from different threads.
3. **Mine runs in a child process** — fully isolated, no locking conflict.
4. **MCP server runs in a child process** — it manages its own client cache
   with auto-reconnect on `chroma.sqlite3` modification (mtime change).

If the GUI needs background search, use a single `asyncio` task or a
`threading.Lock`-guarded queue feeding into a single search thread.

---

## 7. macOS ARM64 / .app Packaging Checklist

| Item | Status | Notes |
|------|--------|-------|
| ChromaDB ≥ 1.5.4 | ✅ pinned | No hnswlib; pure Rust + ONNX |
| ONNX Runtime | ⚠️ CPU only | ARM64 uses `onnxruntime` CPU EP (no ANE/CoreML). Performance is acceptable for embedding (embedding is done offline). |
| Native binaries to bundle | `chromadb/*.so`, `onnxruntime/*.so` | py2app / PyInstaller must include these explicitly in `includes` or `binaries` |
| Palace path | Default `~/.mempalace/palace` | Set `MEMPALACE_PALACE_PATH` env or pass `palace_path=` to adapter to override for sandboxed `.app` |
| SQLite | Built into Python stdlib | ✅ no extra bundling needed |
| Posthog telemetry | Silenced in `__init__.py` | Logger set to CRITICAL; no network calls during tests |
| `fcntl.flock` | Available on macOS | ✅ Used by `mine_lock()` for per-file mining locks |
| Homebrew Python vs system Python | N/A | Bundle a Python interpreter inside the `.app` (standard py2app/PyInstaller approach) |

---

## 8. Rollback Plan

If the 1.5.x upgrade causes unexpected issues in production:

```bash
# Step 1: Stop all MemPalace processes
pkill -f mempalace

# Step 2: Restore palace backup (if created before upgrade)
cp -r ~/.mempalace/palace.bak ~/.mempalace/palace

# Step 3: Downgrade chromadb
pip install 'chromadb==0.6.3' 'chroma-hnswlib==0.7.6'

# Step 4: In pyproject.toml, revert to:
#   "chromadb>=0.5.0"
# and re-lock:
#   uv lock

# Step 5: Verify
python -c "import chromadb; print(chromadb.__version__)"
python -c "from mempalace.gui_adapter import MemPalaceAdapter; print(MemPalaceAdapter.check_chromadb_version())"
```

Note: After rolling back, macOS ARM64 segfaults will return. The rollback is
only intended for x86_64 or Linux environments while a root cause is
investigated.

---

## 9. Next Steps

1. **Lock file update**: Run `uv lock` to regenerate `uv.lock` with
   `chromadb>=1.5.4` (requires `uv` CLI; skipped in sandbox as uv is not
   installed, but `pip install chromadb>=1.5.4` works directly).
2. **macOS CI**: Add a GitHub Actions job on `macos-14` (ARM64) running
   `python -m pytest tests/test_gui_adapter.py` to catch regressions.
3. **GUI integration**: Use `MemPalaceAdapter` as the backend bridge from
   Swift/Flutter. All methods return structured dataclasses with no stdout side
   effects.
4. **Concurrent search**: If the GUI needs background search, implement a
   `SearchWorker` thread that serializes calls through a `queue.Queue`.
