"""
FastAPI Dependencies for Authentication and Model Resource Injection.
"""
import logging
from typing import Optional
from fastapi import Header, HTTPException, Query, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from ..model.config import config
from ..model.model_loader import get_model_manager, GloveModelManager
from ..model.inference import HandDetector

logger = logging.getLogger("backend.api.dependencies")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


def verify_api_key(
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query)
) -> bool:
    """
    Validates API key against configured key.
    If no API key is configured in the environment, allows requests.
    """
    expected_key = config.API_KEY
    if not expected_key:
        return True

    provided_key = header_key or query_key
    if provided_key == expected_key:
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key. Provide 'X-API-Key' header or 'api_key' query parameter."
    )


def get_detector() -> HandDetector:
    """Returns a HandDetector instance backed by the shared model manager."""
    return HandDetector()

