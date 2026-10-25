import logging

from colorama import Fore

from fraud_detection.utils.colored_logging import ColoredFormatter, configure_logging, get_logger


def make_record(level: int, message: str) -> logging.LogRecord:
    return logging.LogRecord("fraud_detection.test", level, __file__, 1, message, None, None)


def test_formatter_colors_level_names():
    formatter = ColoredFormatter()

    assert Fore.GREEN in formatter.format(make_record(logging.INFO, "hello"))
    assert Fore.RED in formatter.format(make_record(logging.ERROR, "boom"))
    assert "boom" in formatter.format(make_record(logging.ERROR, "boom"))


def test_configure_logging_writes_plain_text_file(tmp_path):
    log_file = tmp_path / "logs" / "training.log"
    configure_logging("DEBUG", log_file)

    get_logger("fraud_detection.test").debug("debug message")

    content = log_file.read_text(encoding="utf-8")
    assert "DEBUG" in content
    assert "debug message" in content
    assert "\x1b[" not in content


def test_reconfiguring_does_not_duplicate_handlers():
    configure_logging("INFO")
    handler_count = len(logging.getLogger().handlers)
    configure_logging("WARNING")

    assert len(logging.getLogger().handlers) == handler_count
    assert logging.getLogger().level == logging.WARNING
