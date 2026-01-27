"""Structured logging configuration using structlog."""

import logging
import sys

import structlog

from app.config import get_settings


def _redact_phi_processor(
    logger: structlog.typing.WrappedLogger,
    method_name: str,
    event_dict: structlog.typing.EventDict,
) -> structlog.typing.EventDict:
    """
    Redact PHI (Protected Health Information) from log events.

    Per Spec 0004 security requirements:
    - NEVER log full embedding content or client health data
    - Redact any PII from error messages before logging
    - Use explicit allow-list of loggable fields
    """
    # Fields that may contain PHI and should be redacted
    phi_fields = [
        "content",
        "message_content",
        "response",
        "embedding",
        "content_text",
        "client_message",
        "rationale",
        "body",
        "request_body",
        "text",
        "full_text",
        "transcript",
        "summary",
        "ai_summary",
        "executive_summary",
        "profile_text",
        "query_text",
    ]

    for field in phi_fields:
        if field in event_dict:
            # Check if it's a string and has content
            value = event_dict[field]
            if isinstance(value, str) and len(value) > 0:
                event_dict[field] = "[REDACTED]"
            elif isinstance(value, (list, dict)):
                event_dict[field] = "[REDACTED]"

    return event_dict


def setup_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()

    # Determine processors based on environment
    if settings.is_production:
        # Production: JSON output for log aggregation with PHI redaction
        processors: list[structlog.typing.Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_phi_processor,  # PHI redaction before output
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Pretty console output (PHI redaction optional in dev)
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_phi_processor,  # Also redact in dev for consistency
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(settings.log_level),
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a logger instance with optional name binding."""
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return logger
