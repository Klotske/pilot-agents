"""upsource-gateway: a lightweight client for the Upsource RPC API."""

from . import models
from .client import UpsourceClient
from .errors import (
    AccessDeniedError,
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    UpsourceError,
    UpsourceRpcError,
    UpsourceTransportError,
)

__all__ = [
    "models",
    "UpsourceClient",
    "UpsourceError",
    "UpsourceRpcError",
    "UpsourceTransportError",
    "AuthenticationError",
    "AccessDeniedError",
    "NotFoundError",
    "InvalidRequestError",
]

__version__ = "0.1.0"
