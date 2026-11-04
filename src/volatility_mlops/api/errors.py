"""Structured JSON error handling for the serving API (build-plan section 8.2).

Every error leaves the API as JSON with a stable shape, ``{"error": str,
"details": dict}`` and the right HTTP status, so a client (or the React dashboard)
never has to parse an HTML stack trace. :class:`ApiError` is raised deliberately by
route and service code for client mistakes (unknown ticker, bad date, no model
loaded); unexpected exceptions become a generic 500 without leaking internals.
"""

from __future__ import annotations

from typing import Any

from flask import Flask, Response, jsonify
from werkzeug.exceptions import HTTPException


class ApiError(Exception):
    """An error meant to reach the client as a structured JSON response.

    Attributes:
        status_code: HTTP status to return (e.g. 400, 422, 503).
        message: Human-readable summary placed under ``error``.
        details: Optional machine-readable context placed under ``details``.
    """

    def __init__(
        self, status_code: int, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.details = details or {}

    def to_response(self) -> tuple[Response, int]:
        """Render the error as a ``(json, status)`` tuple."""
        return jsonify({"error": self.message, "details": self.details}), self.status_code


def register_error_handlers(app: Flask) -> None:
    """Install JSON handlers for ApiError, HTTP errors (404/405/...), and crashes."""

    @app.errorhandler(ApiError)
    def _handle_api_error(exc: ApiError) -> tuple[Response, int]:
        return exc.to_response()

    @app.errorhandler(HTTPException)
    def _handle_http_error(exc: HTTPException) -> tuple[Response, int]:
        return (
            jsonify({"error": exc.name, "details": {"description": exc.description}}),
            exc.code or 500,
        )

    @app.errorhandler(Exception)
    def _handle_unexpected(exc: Exception) -> tuple[Response, int]:  # noqa: ARG001
        app.logger.exception("unhandled error")
        return jsonify({"error": "Internal server error", "details": {}}), 500
