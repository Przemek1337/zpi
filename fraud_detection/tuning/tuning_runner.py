from __future__ import annotations

import json
import multiprocessing
import queue
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import yaml
from optuna.trial import FrozenTrial, TrialState

from fraud_detection.configuration.experiment_config import ExperimentConfig
from fraud_detection.exceptions import FraudDetectionError, TuningError
from fraud_detection.training.training_history import TrainingHistory
from fraud_detection.training.training_runner import TrainingRunner
from fraud_detection.tuning.preflight import Preflight, PreflightReport
from fraud_detection.tuning.study_factory import count_trials, open_study
from fraud_detection.tuning.trial_objective import TrialObjective, final_config, quiet_training_logs
from fraud_detection.tuning.tuning_config import TuningConfig
from fraud_detection.utils.colored_logging import configure_logging, get_logger
from fraud_detection.utils.device_resolver import resolve_device

logger = get_logger(__name__)

# errors that fail a single trial (bad hyperparameter combination) without stopping the search
TRIAL_ERRORS = (FraudDetectionError, RuntimeError, ValueError)
MAX_CONSECUTIVE_FAILURES = 5
BANNER_WIDTH = 78


class TuningRunner:
    """Preflight, hyperparameter search across worker processes, result export and final training."""

    def __init__(self, tuning: TuningConfig, base: ExperimentConfig, log_level: str, log_file: Path) -> None:
        self._tuning = tuning
        self._base = base
        self._log_level = log_level
        self._log_file = log_file

    def run(self) -> Optional[TrainingHistory]:
        report, train, validation, device = Preflight(self._base, self._tuning).run()
        deadline = time.time() + self._tuning.budget.timeout_hours * 3600
        if self._tuning.budget.parallel_workers == 1:
            self._announce_start(report, deadline)
            objective = TrialObjective(self._base, self._tuning, train, validation, device)
            _optimize(open_study(self._tuning), objective, self._tuning, deadline, worker_index=0)
        else:
            self._run_workers(report, deadline)

        study = open_study(self._tuning)
        best = self._best_trial(study)
        winner = final_config(self._base, self._tuning, best)
        self._export(study, best, winner)
        history = None
        if self._tuning.final_training.enabled:
            logger.info("Training final model with the best hyperparameters (trial #%d)", best.number)
            history = TrainingRunner(winner).run()
        self._announce_finish(study, best, history)
        return history

    def _run_workers(self, report: PreflightReport, deadline: float) -> None:
        context = multiprocessing.get_context("spawn")  # CUDA cannot be shared with forked processes
        ready = context.Queue()
        workers = [
            context.Process(
                target=_worker_main,
                # configs travel as plain data: parametrised generic pydantic models cannot be pickled
                args=(
                    index,
                    self._tuning.model_dump(mode="json"),
                    self._base.model_dump(mode="json"),
                    deadline,
                    self._log_level,
                    self._log_file,
                    ready,
                ),
                name=f"tuning-worker-{index}",
            )
            for index in range(self._tuning.budget.parallel_workers)
        ]
        for worker in workers:
            worker.start()
        try:
            self._wait_until_ready(workers, ready)
            self._announce_start(report, deadline)
            for worker in workers:
                worker.join()
        except BaseException:
            _stop(workers)
            raise
        failed = [worker.name for worker in workers if worker.exitcode != 0]
        if failed:
            logger.error("Worker(s) %s exited with errors - see log above", ", ".join(failed))

    def _wait_until_ready(self, workers: List[multiprocessing.Process], ready: "queue.Queue") -> None:
        timeout = self._tuning.preflight.worker_start_timeout_seconds
        give_up_at = time.time() + timeout
        started = set()
        while len(started) < len(workers):
            try:
                index, error = ready.get(timeout=1)
            except queue.Empty:
                dead = [worker.name for worker in workers if worker.exitcode is not None]
                if dead:
                    raise TuningError(f"Worker(s) {', '.join(dead)} exited during start-up")
                if time.time() > give_up_at:
                    raise TuningError(f"Workers did not start within {timeout}s")
                continue
            if error is not None:
                raise TuningError(f"Worker {index} failed to start: {error}")
            started.add(index)
        logger.info("[ OK ] %-11s %d running", "Workers", len(workers))

    @staticmethod
    def _best_trial(study: optuna.Study) -> FrozenTrial:
        counts = count_trials(study)
        if not counts["complete"]:
            raise TuningError(
                f"No trial completed ({counts['pruned']} pruned, {counts['fail']} failed); "
                "check the log for trial errors or extend budget.timeout_hours"
            )
        return study.best_trial

    def _export(self, study: optuna.Study, best: FrozenTrial, winner: ExperimentConfig) -> None:
        directory = self._tuning.output.directory
        directory.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe(attrs=("number", "value", "state", "duration", "params", "user_attrs")).to_csv(
            directory / "trials.csv", index=False
        )
        (directory / "best_trial.json").write_text(
            json.dumps(
                {
                    "number": best.number,
                    "validation_fraud_recall": best.value,
                    "max_false_positive_rate": self._base.evaluation.max_false_positive_rate,
                    "params": best.params,
                    "user_attrs": best.user_attrs,
                    "trial_counts": count_trials(study),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (directory / "best_config.yaml").write_text(
            yaml.safe_dump(winner.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        self._export_importances(study, directory)
        self._log_top_trials(study)
        logger.info("Search results written to %s", directory)

    @staticmethod
    def _export_importances(study: optuna.Study, directory: Path) -> None:
        try:
            importances = optuna.importance.get_param_importances(study)
        except (RuntimeError, ValueError) as error:  # too few completed trials to estimate
            logger.warning("Skipping parameter importances: %s", error)
            return
        (directory / "param_importances.json").write_text(json.dumps(importances, indent=2), encoding="utf-8")
        logger.info(
            "Most influential hyperparameters: %s",
            ", ".join(f"{name} {share:.0%}" for name, share in list(importances.items())[:5]),
        )

    def _log_top_trials(self, study: optuna.Study) -> None:
        completed = sorted(
            study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), key=lambda t: t.value, reverse=True
        )
        logger.info("Top %d trials by validation recall:", min(len(completed), self._tuning.output.top_trials))
        for trial in completed[: self._tuning.output.top_trials]:
            logger.info("  #%-4d %s", trial.number, describe_trial(trial))

    def _announce_start(self, report: PreflightReport, deadline: float) -> None:
        finish = datetime.fromtimestamp(deadline)
        lines = [
            "ALL CHECKS PASSED - hyperparameter search is running",
            f"Objective: validation fraud recall at FPR < {self._base.evaluation.max_false_positive_rate:.0%}",
            f"Data: train {report.train_rows} legit | validation {report.validation_rows} "
            f"({report.validation_fraud} fraud) | {report.feature_count} features",
            f"Device: {report.device} ({report.device_name}) | workers: {self._tuning.budget.parallel_workers}",
            f"Search ends around {finish:%Y-%m-%d %H:%M} (+ the trials still running at that point)",
            "Then: final training of the best configuration "
            + ("(enabled)" if self._tuning.final_training.enabled else "(disabled)"),
            f"Results: {self._tuning.output.directory} | log: {self._log_file}",
            "Safe to detach now: Ctrl+B then D (tmux). Re-run the same command to resume.",
        ]
        if report.existing_trials["complete"] or report.existing_trials["pruned"]:
            lines.insert(1, f"Resuming study with {sum(report.existing_trials.values())} earlier trials")
        _banner(lines)

    def _announce_finish(self, study: optuna.Study, best: FrozenTrial, history: Optional[TrainingHistory]) -> None:
        counts = count_trials(study)
        lines = [
            "TUNING FINISHED",
            f"Trials: {counts['complete']} complete, {counts['pruned']} pruned, {counts['fail']} failed",
            f"Best trial #{best.number}: validation recall {best.value:.1%}",
            f"  {describe_trial(best)}",
            f"Best config: {self._tuning.output.directory / 'best_config.yaml'}",
        ]
        if history is not None and history.best_record is not None:
            recall = history.best_record.validation_fraud_recall
            lines.append(
                f"Final model: validation recall {recall:.1%} -> {self._tuning.output.directory / 'final_model'}"
            )
        lines.append("Next: evaluate the final model once on the test set (separate step).")
        _banner(lines)


def describe_trial(trial: FrozenTrial) -> str:
    params, attrs = trial.params, trial.user_attrs
    parts = [
        f"recall {trial.value:.1%}" if trial.value is not None else trial.state.name.lower(),
        f"layers {attrs.get('hidden_dims')} latent {params.get('latent_dim')}",
        f"{params.get('activation')} bn={params.get('batch_norm')} dropout {params.get('dropout', 0):.2f}",
        f"{params.get('optimizer')} lr {params.get('learning_rate', 0):.1e} wd {params.get('weight_decay', 0):.1e}",
        f"bs {params.get('batch_size')}",
        f"noise {params['input_noise_std']:.3f}" if params.get("denoising") else "noise off",
    ]
    if "best_epoch" in attrs:
        parts.append(f"best ep {attrs['best_epoch']}/{attrs['epochs_run']} | {attrs['minutes']} min")
    return " | ".join(parts)


class _TrialLogger:
    def __init__(self, worker_index: int) -> None:
        self._prefix = f"[worker {worker_index}]"
        self._consecutive_failures = 0

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state == TrialState.COMPLETE:
            self._consecutive_failures = 0
            best = study.best_trial
            logger.info(
                "%s Trial #%d %s || best %.1f%% (#%d)",
                self._prefix, trial.number, describe_trial(trial), best.value * 100, best.number,
            )
        elif trial.state == TrialState.PRUNED:
            self._consecutive_failures = 0
            last = trial.intermediate_values.get(trial.last_step) if trial.last_step is not None else None
            logger.info(
                "%s Trial #%d pruned at epoch %s (recall %s)",
                self._prefix, trial.number, trial.last_step, "n/a" if last is None else f"{last:.1%}",
            )
        elif trial.state == TrialState.FAIL:
            self._consecutive_failures += 1
            logger.warning("%s Trial #%d failed with %s", self._prefix, trial.number, trial.params)
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "%s %d trials failed in a row - stopping this worker", self._prefix, self._consecutive_failures
                )
                study.stop()


def _optimize(
    study: optuna.Study, objective: TrialObjective, tuning: TuningConfig, deadline: float, worker_index: int
) -> None:
    remaining = deadline - time.time()
    if remaining <= 0:
        logger.warning("Time budget already used up - no new trials")
        return
    callbacks = [_TrialLogger(worker_index)]
    if tuning.budget.max_trials is not None:
        callbacks.append(
            optuna.study.MaxTrialsCallback(tuning.budget.max_trials, states=(TrialState.COMPLETE, TrialState.PRUNED))
        )
    study.optimize(objective, timeout=remaining, catch=TRIAL_ERRORS, callbacks=callbacks, gc_after_trial=True)


def _worker_main(
    index: int,
    raw_tuning: Dict[str, Any],
    raw_base: Dict[str, Any],
    deadline: float,
    log_level: str,
    log_file: Path,
    ready: "multiprocessing.Queue",
) -> None:
    configure_logging(log_level, log_file)
    try:
        tuning = TuningConfig.model_validate(raw_tuning)
        base = ExperimentConfig.model_validate(raw_base)
        device = resolve_device(base.device)
        with quiet_training_logs():
            train, validation = TrainingRunner(base).load_datasets()
        study = open_study(tuning, worker_index=index)
        objective = TrialObjective(base, tuning, train, validation, device)
    except Exception as error:
        ready.put((index, f"{type(error).__name__}: {error}"))
        sys.exit(1)
    ready.put((index, None))
    try:
        _optimize(study, objective, tuning, deadline, index)
    except KeyboardInterrupt:
        sys.exit(130)


def _stop(workers: List[multiprocessing.Process]) -> None:
    for worker in workers:
        worker.join(timeout=30)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)


def _banner(lines: List[str]) -> None:
    logger.info("=" * BANNER_WIDTH)
    for line in lines:
        logger.info(line)
    logger.info("=" * BANNER_WIDTH)
