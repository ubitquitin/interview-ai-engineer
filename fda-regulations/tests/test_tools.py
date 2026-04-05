"""
Unit tests for RAG tool.

Uses mocks to avoid ChromaDB/embedding dependencies.
Tests search logic without requiring vector database.
"""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

# Mock ChromaDB components BEFORE importing to prevent initialization errors
sys.modules['chromadb'] = MagicMock()

with patch('langchain_community.vectorstores.Chroma', MagicMock()):
    with patch('langchain_community.embeddings.fastembed.FastEmbedEmbeddings', MagicMock()):
        from src.tools.rag_tool import search_fda_precedents


class MockDocument:
    """Mock LangChain Document object."""
    def __init__(self, page_content: str):
        self.page_content = page_content


class TestSearchFDAPrecedents:
    """Test suite for FDA precedent search tool."""

    @patch('src.tools.rag_tool.vector_db')
    def test_search_returns_top_results(self, mock_vector_db):
        """Test that search returns top 3 results."""
        # Mock similarity search response
        mock_docs = [
            MockDocument("Deficiency 1: Cleaning validation failure per 21 CFR 211.67"),
            MockDocument("Deficiency 2: Inadequate batch records per 21 CFR 211.188"),
            MockDocument("Deficiency 3: Missing audit trails per 21 CFR 211.194"),
        ]
        mock_vector_db.similarity_search.return_value = mock_docs

        # Execute search
        results = search_fda_precedents("cleaning validation")

        # Verify
        assert len(results) == 3
        assert "Cleaning validation" in results[0]
        assert "batch records" in results[1]
        assert "audit trails" in results[2]
        mock_vector_db.similarity_search.assert_called_once_with("cleaning validation", k=3)

    @patch('src.tools.rag_tool.vector_db')
    def test_search_with_empty_query(self, mock_vector_db):
        """Test search with empty query string."""
        mock_vector_db.similarity_search.return_value = []

        results = search_fda_precedents("")

        assert results == []
        mock_vector_db.similarity_search.assert_called_once_with("", k=3)

    @patch('src.tools.rag_tool.vector_db')
    def test_search_with_cfr_reference(self, mock_vector_db):
        """Test search with CFR reference number."""
        mock_docs = [
            MockDocument("21 CFR 211.194 - Laboratory records must include complete data"),
        ]
        mock_vector_db.similarity_search.return_value = mock_docs

        results = search_fda_precedents("21 CFR 211.194 laboratory records")

        assert len(results) == 1
        assert "21 CFR 211.194" in results[0]
        assert "Laboratory records" in results[0]

    @patch('src.tools.rag_tool.vector_db')
    def test_search_extracts_page_content(self, mock_vector_db):
        """Test that only page_content is extracted from documents."""
        mock_docs = [
            MockDocument("Content 1"),
            MockDocument("Content 2"),
        ]
        mock_vector_db.similarity_search.return_value = mock_docs

        results = search_fda_precedents("test query")

        # Should only return page_content strings
        assert all(isinstance(r, str) for r in results)
        assert results == ["Content 1", "Content 2"]

    @patch('src.tools.rag_tool.vector_db')
    def test_search_handles_complex_query(self, mock_vector_db):
        """Test search with multi-term complex query."""
        mock_docs = [
            MockDocument("Sterile manufacturing cleaning validation deficiency"),
        ]
        mock_vector_db.similarity_search.return_value = mock_docs

        query = "21 CFR 211 cleaning validation sterile manufacturing"
        results = search_fda_precedents(query)

        assert len(results) == 1
        mock_vector_db.similarity_search.assert_called_once_with(query, k=3)
