from __future__ import annotations

import logging

try:
    from rich.logging import RichHandler

    RICH_LOGGING = True
except ImportError:
    RichHandler = None
    RICH_LOGGING = False


if RICH_LOGGING:

    class AquilesRichHandler(RichHandler):
        def emit(self, record: logging.LogRecord) -> None:
            if not record.args:
                record.msg = f"[dim][{record.name}][/dim] {record.msg}"
                self.markup = True
            super().emit(record)


def get_rich_handler(level: int = logging.INFO):
    if RICH_LOGGING:
        return AquilesRichHandler(
            level=level,
            show_path=False,
            show_time=False,
            rich_tracebacks=True,
            markup=True,
        )
    return None


def uvicorn_log_config() -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "rich": {
                "class": "rich.logging.RichHandler",
                "show_path": False,
                "show_time": False,
                "rich_tracebacks": True,
                "markup": True,
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["rich"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["rich"], "level": "INFO", "propagate": False},
        },
    }
