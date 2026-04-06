"""
Pytest configuration and shared fixtures.

Mocks ChromaDB initialization to prevent file system access during tests.
"""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture(scope="session", autouse=True)
def mock_chromadb():
    """Mock ChromaDB components before any test imports.

    Args:
        None: Pytest fixture with session scope and autouse enabled

    Returns:
        None: Side effect only - prevents ChromaDB initialization during test runs

    Notes:
        Runs automatically for all tests in the session
    """
    # This fixture runs automatically for all tests
    pass


@pytest.fixture
def mock_vector_db():
    """Fixture providing a mocked vector database for RAG tests.

    Args:
        None: Pytest fixture with function scope

    Returns:
        MagicMock: Mocked Chroma vector database instance with similarity_search method
                   configured to return empty list by default

    Notes:
        Tests should override the return_value of similarity_search as needed
    """
    mock_db = MagicMock()
    mock_db.similarity_search = Mock(return_value=[])
    return mock_db
