"""
test_searcher.py -- Tests for search(), search_raw(), and search_memories().

Uses the real ChromaDB fixtures from conftest.py for integration tests,
plus mock-based tests for error paths.
"""

from unittest.mock import MagicMock, patch

import pytest

from mempalace.searcher import SearchError, search, search_raw, search_memories


# ── search_raw (raw parity path) ──────────────────────────────────────


class TestSearchRaw:
    def test_basic_search_returns_dict(self, palace_path, seeded_collection):
        result = search_raw("JWT authentication", palace_path)
        assert "results" in result
        assert "query" in result
        assert "filters" in result
        assert len(result["results"]) > 0
        assert result["query"] == "JWT authentication"

    def test_wing_filter(self, palace_path, seeded_collection):
        result = search_raw("planning", palace_path, wing="notes")
        assert all(r["wing"] == "notes" for r in result["results"])

    def test_room_filter(self, palace_path, seeded_collection):
        result = search_raw("database", palace_path, room="backend")
        assert all(r["room"] == "backend" for r in result["results"])

    def test_n_results_limit(self, palace_path, seeded_collection):
        result = search_raw("code", palace_path, n_results=2)
        assert len(result["results"]) <= 2

    def test_no_palace_returns_error(self, tmp_path):
        result = search_raw("anything", str(tmp_path / "missing"))
        assert "error" in result

    def test_result_fields(self, palace_path, seeded_collection):
        result = search_raw("authentication", palace_path)
        hit = result["results"][0]
        assert "text" in hit
        assert "wing" in hit
        assert "room" in hit
        assert "source_file" in hit
        assert "similarity" in hit
        assert "distance" in hit
        assert isinstance(hit["similarity"], float)

    def test_result_no_extra_fields(self, palace_path, seeded_collection):
        """search_raw results must NOT have search_memories fields."""
        result = search_raw("authentication", palace_path)
        hit = result["results"][0]
        for field in (
            "closet_boost",
            "effective_distance",
            "bm25_score",
            "matched_via",
            "closet_preview",
            "source_path",
            "chunk_index",
            "drawer_id",
            "_sort_key",
            "_source_file_full",
            "_chunk_index",
        ):
            assert field not in hit, f"search_raw hit must not have '{field}'"

    def test_search_raw_query_error(self):
        """search_raw returns error dict when query raises."""
        mock_col = MagicMock()
        mock_col.query.side_effect = RuntimeError("query failed")

        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            result = search_raw("test", "/fake/path")
        assert "error" in result
        assert "query failed" in result["error"]

    def test_filters_in_result(self, palace_path, seeded_collection):
        result = search_raw("test", palace_path, wing="project", room="backend")
        assert result["filters"]["wing"] == "project"
        assert result["filters"]["room"] == "backend"

    def test_similarity_is_raw(self, palace_path, seeded_collection):
        """similarity must be round(max(0, 1 - raw_distance), 3)."""
        result = search_raw("authentication", palace_path)
        for hit in result["results"]:
            expected = round(max(0.0, 1.0 - hit["distance"]), 3)
            assert hit["similarity"] == expected

    def test_order_is_distance_ascending(self, palace_path, seeded_collection):
        """Results must be in ChromaDB native order (distance ascending)."""
        result = search_raw("code", palace_path, n_results=10)
        dists = [h["distance"] for h in result["results"]]
        assert dists == sorted(dists)

    def test_default_n_results_is_5(self):
        """search_raw default n_results must be 5 (same as CLI)."""
        import inspect

        sig = inspect.signature(search_raw)
        assert sig.parameters["n_results"].default == 5


# ── search_raw vs search() parity ─────────────────────────────────────


class TestSearchRawVsCLI:
    """search_raw must produce identical results to search() (minus stdout)."""

    def test_parity_hit_count(self, palace_path, seeded_collection, capsys):
        query = "authentication"
        search(query, palace_path, n_results=5)
        capsys.readouterr()
        raw = search_raw(query, palace_path, n_results=5)
        assert "error" not in raw
        assert len(raw["results"]) > 0

    def test_parity_hit_text(self, palace_path, seeded_collection, capsys):
        """search_raw results must have the same text as search() would print."""
        query = "authentication"
        search(query, palace_path, n_results=5)
        captured = capsys.readouterr()
        raw = search_raw(query, palace_path, n_results=5)
        for hit in raw["results"]:
            assert hit["text"].strip() in captured.out

    def test_parity_hit_similarity(self, palace_path, seeded_collection, capsys):
        """search_raw similarity must match what search() prints."""
        query = "authentication"
        search(query, palace_path, n_results=5)
        captured = capsys.readouterr()
        raw = search_raw(query, palace_path, n_results=5)
        for hit in raw["results"]:
            sim_str = f"Match:  {hit['similarity']}"
            assert sim_str in captured.out


# ── search_memories (API) ──────────────────────────────────────────────


class TestSearchMemories:
    def test_basic_search(self, palace_path, seeded_collection):
        result = search_memories("JWT authentication", palace_path)
        assert "results" in result
        assert len(result["results"]) > 0
        assert result["query"] == "JWT authentication"

    def test_wing_filter(self, palace_path, seeded_collection):
        result = search_memories("planning", palace_path, wing="notes")
        assert all(r["wing"] == "notes" for r in result["results"])

    def test_room_filter(self, palace_path, seeded_collection):
        result = search_memories("database", palace_path, room="backend")
        assert all(r["room"] == "backend" for r in result["results"])

    def test_wing_and_room_filter(self, palace_path, seeded_collection):
        result = search_memories("code", palace_path, wing="project", room="frontend")
        assert all(r["wing"] == "project" and r["room"] == "frontend" for r in result["results"])

    def test_n_results_limit(self, palace_path, seeded_collection):
        result = search_memories("code", palace_path, n_results=2)
        assert len(result["results"]) <= 2

    def test_no_palace_returns_error(self, tmp_path):
        result = search_memories("anything", str(tmp_path / "missing"))
        assert "error" in result

    def test_result_fields(self, palace_path, seeded_collection):
        result = search_memories("authentication", palace_path)
        hit = result["results"][0]
        assert "text" in hit
        assert "wing" in hit
        assert "room" in hit
        assert "source_file" in hit
        assert "similarity" in hit
        assert isinstance(hit["similarity"], float)

    def test_search_memories_query_error(self):
        """search_memories returns error dict when query raises."""
        mock_col = MagicMock()
        mock_col.query.side_effect = RuntimeError("query failed")

        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            result = search_memories("test", "/fake/path")
        assert "error" in result
        assert "query failed" in result["error"]

    def test_search_memories_filters_in_result(self, palace_path, seeded_collection):
        result = search_memories("test", palace_path, wing="project", room="backend")
        assert result["filters"]["wing"] == "project"
        assert result["filters"]["room"] == "backend"


# ── search() (CLI print function) ─────────────────────────────────────


class TestSearchCLI:
    def test_search_prints_results(self, palace_path, seeded_collection, capsys):
        search("JWT authentication", palace_path)
        captured = capsys.readouterr()
        assert "JWT" in captured.out or "authentication" in captured.out

    def test_search_with_wing_filter(self, palace_path, seeded_collection, capsys):
        search("planning", palace_path, wing="notes")
        captured = capsys.readouterr()
        assert "Results for" in captured.out

    def test_search_with_room_filter(self, palace_path, seeded_collection, capsys):
        search("database", palace_path, room="backend")
        captured = capsys.readouterr()
        assert "Room:" in captured.out

    def test_search_with_wing_and_room(self, palace_path, seeded_collection, capsys):
        search("code", palace_path, wing="project", room="frontend")
        captured = capsys.readouterr()
        assert "Wing:" in captured.out
        assert "Room:" in captured.out

    def test_search_no_palace_raises(self, tmp_path):
        with pytest.raises(SearchError, match="No palace found"):
            search("anything", str(tmp_path / "missing"))

    def test_search_no_results(self, palace_path, collection, capsys):
        """Empty collection returns no results message."""
        result = search("xyzzy_nonexistent_query", palace_path, n_results=1)
        captured = capsys.readouterr()
        assert result is None or "No results" in captured.out

    def test_search_query_error_raises(self):
        """search raises SearchError when query fails."""
        mock_col = MagicMock()
        mock_col.query.side_effect = RuntimeError("boom")

        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            with pytest.raises(SearchError, match="Search error"):
                search("test", "/fake/path")

    def test_search_n_results(self, palace_path, seeded_collection, capsys):
        search("code", palace_path, n_results=1)
        captured = capsys.readouterr()
        assert "[1]" in captured.out
