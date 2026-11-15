"""Exception hierarchy for the Upsource RPC client."""

from __future__ import annotations

from .models import RPCErrors

# code -> (http_status, subclass factory)
_HTTP_STATUS: dict[int, int] = {
    RPCErrors.OK: 200,
    RPCErrors.Unknown: 500,
    RPCErrors.InvalidRequest: 400,
    RPCErrors.AuthenticationError: 401,
    RPCErrors.AccessDenied: 401,
    RPCErrors.SecurityTokenExpired: 419,
    RPCErrors.SecurityTokenRequired: 401,
    RPCErrors.RequestRejected: 403,
    RPCErrors.NotFound: 404,
    RPCErrors.MethodNotSupported: 405,
    RPCErrors.InternalError: 500,
    RPCErrors.TaskCancelled: 204,
    RPCErrors.ServerReadOnly: 503,
    RPCErrors.NotSupportedInProduction: 501,
    RPCErrors.UpsourceException: 500,
    RPCErrors.FileNotFoundInRevision: 404,
    RPCErrors.RevisionNotIndexed: 400,
    RPCErrors.UpsourceNotReadyException: 503,
    RPCErrors.FileIsTooBig: 413,
}


class UpsourceError(Exception):
    """Base class for every error raised by this package."""


class UpsourceTransportError(UpsourceError):
    """A network-level or HTTP-level failure (timeout, connection, 5xx)."""


class UpsourceRpcError(UpsourceError):
    """A structured RPC error returned by Upsource: ``{"error": {"code", "message"}}``."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        try:
            self.name = RPCErrors(code).name
        except ValueError:
            self.name = "Unknown"
        self.http_status = _HTTP_STATUS.get(code, 500)
        super().__init__(f"[{self.name} ({code})] {message}")


class InvalidRequestError(UpsourceRpcError):
    pass


class AuthenticationError(UpsourceRpcError):
    pass


class AccessDeniedError(UpsourceRpcError):
    pass


class NotFoundError(UpsourceRpcError):
    pass


# code -> subclass; unknown codes fall back to the base UpsourceRpcError.
_CODE_TO_ERROR: dict[int, type[UpsourceRpcError]] = {
    RPCErrors.InvalidRequest: InvalidRequestError,
    RPCErrors.AuthenticationError: AuthenticationError,
    RPCErrors.AccessDenied: AccessDeniedError,
    RPCErrors.NotFound: NotFoundError,
    RPCErrors.FileNotFoundInRevision: NotFoundError,
}


def build_error(error: dict) -> UpsourceRpcError:
    """Construct the right exception from an ``error`` payload dict."""
    code = error.get("code", RPCErrors.Unknown)
    message = error.get("message", "")
    if not isinstance(code, int):
        code = RPCErrors.Unknown
    cls = _CODE_TO_ERROR.get(code, UpsourceRpcError)
    return cls(code, message)
