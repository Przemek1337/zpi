from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from colorama import Fore, Style, just_fix_windows_console

from fraud_detection.exceptions import ConfigurationError

_PLAIN_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MANAGED_HANDLER_FLAG = "_fraud_detection_managed"


class ColoredFormatter(logging.Formatter):
    _LEVEL_COLORS: Dict[int, str] = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def __init__(self) -> None:
        super().__init__(datefmt=_DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLORS.get(record.levelno, "")
        message_color = color if record.levelno >= logging.WARNING else ""
        line = (
            f"{Style.DIM}{self.formatTime(record, self.datefmt)}{Style.RESET_ALL} | "
            f"{color}{Style.BRIGHT}{record.levelname:<8}{Style.RESET_ALL} | "
            f"{Fore.MAGENTA}{record.name}{Style.RESET_ALL} | "
            f"{message_color}{record.getMessage()}{Style.RESET_ALL}"
        )
        if record.exc_info:
            line += f"\n{Fore.RED}{self.formatException(record.exc_info)}{Style.RESET_ALL}"
        return line


def configure_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    just_fix_windows_console()
    root = logging.getLogger()
    reset_logging()
    root.setLevel(level)
    _attach(root, _console_handler())
    if log_file is not None:
        _attach(root, _file_handler(Path(log_file)))


def reset_logging() -> None:
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _MANAGED_HANDLER_FLAG, False)]:
        root.removeHandler(handler)
        handler.close()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _console_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColoredFormatter())
    return handler


def _file_handler(log_file: Path) -> logging.Handler:
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"Cannot open log file {log_file}: {error}") from error
    handler.setFormatter(logging.Formatter(_PLAIN_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _attach(logger: logging.Logger, handler: logging.Handler) -> None:
    setattr(handler, _MANAGED_HANDLER_FLAG, True)
    logger.addHandler(handler)
