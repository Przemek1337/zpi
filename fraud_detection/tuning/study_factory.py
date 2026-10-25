from __future__ import annotations

import warnings
from typing import Dict

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from fraud_detection.exceptions import TuningError
from fraud_detection.tuning.tuning_config import TuningConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def open_study(tuning: TuningConfig, worker_index: int = 0) -> optuna.Study:
    """Creates the study or loads it from the journal file shared by all workers."""
    storage_path = tuning.study.storage
    try:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        # open-based lock works on every OS (the default symlink lock needs extra privileges on Windows)
        backend = JournalFileBackend(str(storage_path), lock_obj=JournalFileOpenLock(str(storage_path)))
        return optuna.create_study(
            storage=JournalStorage(backend),
            study_name=tuning.study.name,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=tuning.study.random_startup_trials,
                multivariate=True,
                group=True,
                constant_liar=True,
                # distinct seeds keep parallel workers from proposing identical random trials
                seed=tuning.study.sampler_seed + worker_index,
            ),
            pruner=_pruner(tuning),
        )
    except OSError as error:
        raise TuningError(f"Cannot open study storage {storage_path}: {error}") from error


def fail_interrupted_trials(study: optuna.Study) -> int:
    """Marks trials left RUNNING by a killed previous run as failed so resuming starts cleanly."""
    stale = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    for trial in stale:
        study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)
    return len(stale)


def count_trials(study: optuna.Study) -> Dict[str, int]:
    counts = {state.name.lower(): 0 for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)}
    for trial in study.get_trials(deepcopy=False):
        key = trial.state.name.lower()
        if key in counts:
            counts[key] += 1
    return counts


def best_trial_or_none(study: optuna.Study):
    return study.best_trial if count_trials(study)["complete"] else None


def _pruner(tuning: TuningConfig) -> optuna.pruners.BasePruner:
    if not tuning.pruning.enabled:
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=tuning.pruning.startup_trials,
        n_warmup_steps=tuning.pruning.warmup_epochs,
    )
