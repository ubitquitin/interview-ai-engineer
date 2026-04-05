"""
Pytest configuration and shared fixtures.

Mocks ChromaDB initialization to prevent file system access during tests.
"""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture(scope="session", autouse=True)
def mock_chromadb():
    """Mock ChromaDB components before any test imports."""
    # This fixture runs automatically for all tests
    pass


@pytest.fixture
def mock_vector_db():
    """Fixture providing a mocked vector database for RAG tests."""
    mock_db = MagicMock()
    mock_db.similarity_search = Mock(return_value=[])
    return mock_db
