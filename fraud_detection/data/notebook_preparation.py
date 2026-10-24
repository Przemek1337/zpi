from __future__ import annotations

import time
from pathlib import Path

from fraud_detection.exceptions import DataLoadingError
from fraud_detection.utils.colored_logging import get_logger

logger = get_logger(__name__)

CELL_TIMEOUT_SECONDS = 1800


def prepare_splits(splits_path: Path, notebook: Path) -> None:
    """Executes the preprocessing notebook headlessly so that it writes ``splits_path``.

    The notebook runs in its own directory (it uses relative paths) and is never modified on disk.
    """
    splits_path, notebook = Path(splits_path), Path(notebook)
    if not notebook.is_file():
        raise DataLoadingError(f"Preparation notebook not found: {notebook.absolute()}")
    nbformat, NotebookClient = _import_notebook_runner()
    logger.info("Splits archive %s missing - executing %s to create it", splits_path, notebook)
    started = time.perf_counter()
    try:
        document = nbformat.read(notebook, as_version=4)
        NotebookClient(
            document,
            timeout=CELL_TIMEOUT_SECONDS,
            kernel_name="python3",
            resources={"metadata": {"path": str(notebook.parent.absolute())}},
        ).execute()
    except Exception as error:  # nbclient raises a wide range of kernel/cell errors
        raise DataLoadingError(f"Executing {notebook} failed: {_first_line(error)}") from error
    if not splits_path.is_file():
        raise DataLoadingError(f"{notebook} finished but did not create {splits_path.absolute()}")
    logger.info("Notebook finished in %.0fs, created %s", time.perf_counter() - started, splits_path)


def _import_notebook_runner():
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as error:
        raise DataLoadingError(
            "Executing the preparation notebook requires nbclient, nbformat and ipykernel "
            "(pip install -r requirements.txt)"
        ) from error
    return nbformat, NotebookClient


def _first_line(error: Exception) -> str:
    text = str(error).strip()
    return text.splitlines()[-1] if text else type(error).__name__
