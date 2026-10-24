import json
import logging
from pathlib import Path

import optuna
import pytest
import torch
import yaml
from pydantic import ValidationError

from fraud_detection.configuration.experiment_config import ExperimentConfig
from fraud_detection.configuration.yaml_config_loader import YamlConfigLoader
from fraud_detection.exceptions import ConfigurationError, DataValidationError, DeviceUnavailableError, TuningError
from fraud_detection.persistence.model_artifact_store import ModelArtifactStore
from fraud_detection.tuning.preflight import Preflight
from fraud_detection.tuning.search_space import SearchSpace, apply_overrides
from fraud_detection.tuning.study_factory import count_trials, fail_interrupted_trials, open_study
from fraud_detection.tuning.trial_objective import TrialObjective, final_config, trial_config
from fraud_detection.tuning.tuning_config import SearchSpaceConfig, TuningConfig
from fraud_detection.tuning.tuning_runner import TuningRunner
from tests.conftest import make_transactions, save_splits
from tune_autoencoder import ExitCode, main

CPU = torch.device("cpu")


@pytest.fixture
def base_dict(tmp_path) -> dict:
    splits = save_splits(
        tmp_path / "splits.npz",
        make_transactions(legitimate=400, fraud=0, seed=1),
        make_transactions(legitimate=150, fraud=15, seed=2),
    )
    return {
        "seed": 7,
        "device": "cpu",
        "data": {"splits_path": str(splits), "drop_columns": ["Time"]},
        "training": {"epochs": 3, "batch_size": 64},
        "output": {"directory": str(tmp_path / "unused")},
    }


@pytest.fixture
def tuning_dict(tmp_path, base_dict) -> dict:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base_dict), encoding="utf-8")
    return {
        "base_config": str(base_path),
        "study": {"storage": str(tmp_path / "tuning" / "study.journal"), "random_startup_trials": 2},
        "budget": {"timeout_hours": 1, "max_trials": 3, "parallel_workers": 1},
        "trial": {"epochs": 2, "early_stopping_patience": 2},
        "pruning": {"startup_trials": 1, "warmup_epochs": 1},
        "preflight": {"require_gpu": False, "smoke_trials": 2, "smoke_epochs": 1, "smoke_train_rows": 128},
        "output": {"directory": str(tmp_path / "tuning")},
        "search_space": {"batch_size": {"choices": [64, 128]}, "latent_dim": {"low": 2, "high": 6}},
    }


def load(tuning_dict):
    tuning = TuningConfig.model_validate(tuning_dict)
    return tuning, YamlConfigLoader().load(tuning.base_config)


def write_tuning(tmp_path, tuning_dict) -> Path:
    path = tmp_path / "tuning.yaml"
    path.write_text(yaml.safe_dump(tuning_dict), encoding="utf-8")
    return path


def test_default_tuning_config_file_is_valid():
    tuning = YamlConfigLoader(TuningConfig).load(Path("config/autoencoder_tuning.yaml"))

    assert tuning.base_config == Path("config/autoencoder_training.yaml")
    assert tuning.search_space == SearchSpaceConfig()


@pytest.mark.parametrize(
    "space",
    [
        {"latent_dim": {"low": 8, "high": 4}},
        {"learning_rate": {"low": 0, "high": 1e-2, "log": True}},
        {"width_shrink": {"low": 0.5, "high": 1.5}},
        {"dropout": {"low": 0.0, "high": 1.0}},
        {"activation": {"choices": ["swish"]}},
        {"batch_size": {"choices": []}},
        {"hidden_layers": {"low": 0, "high": 2}},
        {"unknown": {"low": 1, "high": 2}},
    ],
)
def test_invalid_search_space_is_rejected(space):
    with pytest.raises(ValidationError):
        TuningConfig.model_validate({"search_space": space})


def test_every_sampled_configuration_is_valid(tuning_dict):
    tuning, base = load(tuning_dict)
    tuning = tuning.model_copy(update={"search_space": SearchSpaceConfig()})
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    for _ in range(300):
        trial = study.ask()
        config = trial_config(base, tuning, trial)
        study.tell(trial, 0.0)

        assert min(config.model.hidden_dims) > config.model.latent_dim
        assert config.training.epochs == tuning.trial.epochs
        assert (config.training.input_noise_std > 0) == trial.params["denoising"]
        assert ("scheduler_factor" in trial.params) == trial.params["scheduler_enabled"]


def test_hidden_layers_shrink_towards_bottleneck():
    trial = optuna.trial.FixedTrial(
        {"latent_dim": 6, "hidden_layers": 3, "first_hidden_dim": 64, "width_shrink": 0.3}
    )
    space = SearchSpace(SearchSpaceConfig())

    assert space._hidden_dims(trial, latent_dim=6) == (64, 19, 7)


def test_final_config_rebuilds_the_winning_trial(tuning_dict, base_dict):
    tuning, base = load(tuning_dict)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    trial = study.ask()
    sampled = trial_config(base, tuning, trial)
    study.tell(trial, 0.5)

    final = final_config(base, tuning, study.best_trial)

    assert final.model == sampled.model
    assert final.optimizer == sampled.optimizer
    assert final.scheduler == sampled.scheduler
    assert final.training.batch_size == sampled.training.batch_size
    assert final.training.input_noise_std == sampled.training.input_noise_std
    assert final.training.epochs == base_dict["training"]["epochs"]
    assert final.output.directory == tuning.output.directory / "final_model"


def test_invalid_override_is_a_configuration_error(tuning_dict):
    _, base = load(tuning_dict)

    with pytest.raises(ConfigurationError, match="Sampled configuration is invalid"):
        apply_overrides(base, {"model": {"hidden_dims": [4], "latent_dim": 8}})


def test_objective_returns_recall_and_records_trial_details(tuning_dict):
    tuning, base = load(tuning_dict)
    train, validation = load_data(base)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=1))

    study.optimize(TrialObjective(base, tuning, train, validation, CPU), n_trials=1)

    trial = study.trials[0]
    assert 0.0 <= trial.value <= 1.0
    assert set(trial.intermediate_values) == {1, 2}
    assert {"parameters", "best_epoch", "epochs_run", "val_legit_mse", "hidden_dims"} <= set(trial.user_attrs)


def test_objective_prunes_through_epoch_reports(tuning_dict):
    tuning, base = load(tuning_dict)
    train, validation = load_data(base)

    class AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    study = optuna.create_study(direction="maximize", pruner=AlwaysPrune())
    study.optimize(TrialObjective(base, tuning, train, validation, CPU), n_trials=1)

    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert study.trials[0].user_attrs["pruned_at_epoch"] == 1


def test_interrupted_trials_are_failed_on_resume(tuning_dict):
    tuning, _ = load(tuning_dict)
    study = open_study(tuning)
    study.ask()  # left RUNNING, as after a killed process

    assert fail_interrupted_trials(open_study(tuning)) == 1
    assert count_trials(open_study(tuning))["fail"] == 1


def test_preflight_reports_data_and_study(tuning_dict, caplog):
    caplog.set_level(logging.INFO)
    tuning, base = load(tuning_dict)

    report, train, validation, device = Preflight(base, tuning).run()

    assert (report.train_rows, report.validation_rows, report.validation_fraud) == (400, 165, 15)
    assert report.feature_count == train.feature_count == 29
    assert device == CPU
    assert report.existing_trials == {"complete": 0, "pruned": 0, "fail": 0}
    assert "[ OK ] Smoke test" in caplog.text
    assert not (tuning.output.directory / ".preflight_smoke").exists()


def test_preflight_requires_fraud_in_validation(tuning_dict, base_dict, tmp_path):
    save_splits(Path(base_dict["data"]["splits_path"]), make_transactions(50, 0), make_transactions(20, 0))
    tuning, base = load(tuning_dict)

    with pytest.raises(DataValidationError, match="no fraud"):
        Preflight(base, tuning).run()


def test_preflight_requires_gpu_when_configured(tuning_dict):
    tuning_dict["preflight"]["require_gpu"] = True
    tuning, base = load(tuning_dict)

    if torch.cuda.is_available():
        pytest.skip("CUDA available")
    with pytest.raises(DeviceUnavailableError, match="require_gpu"):
        Preflight(base, tuning).run()


def test_preflight_rejects_latent_wider_than_input(tuning_dict):
    tuning_dict["search_space"]["latent_dim"] = {"low": 2, "high": 40}
    tuning, base = load(tuning_dict)

    with pytest.raises(TuningError, match="latent_dim.high"):
        Preflight(base, tuning).run()


def test_runner_searches_exports_and_trains_final_model(tuning_dict, caplog):
    caplog.set_level(logging.INFO)
    tuning, base = load(tuning_dict)

    history = TuningRunner(tuning, base, "INFO", tuning.output.directory / "tuning.log").run()

    output = tuning.output.directory
    best = json.loads((output / "best_trial.json").read_text(encoding="utf-8"))
    best_config = YamlConfigLoader().load(output / "best_config.yaml")
    model, _ = ModelArtifactStore(output / "final_model").load_model(CPU)
    assert sum(best["trial_counts"].values()) == 3
    assert (output / "trials.csv").is_file()
    assert best_config.output.directory == output / "final_model"
    assert model.config == best_config.model
    assert history.best_record.validation_fraud_recall is not None
    assert "ALL CHECKS PASSED" in caplog.text
    assert "TUNING FINISHED" in caplog.text


def test_runner_resumes_existing_study(tuning_dict):
    tuning_dict["final_training"] = {"enabled": False}
    tuning, base = load(tuning_dict)
    TuningRunner(tuning, base, "INFO", tuning.output.directory / "tuning.log").run()

    resumed = tuning.model_copy(update={"budget": tuning.budget.model_copy(update={"max_trials": 5})})
    TuningRunner(resumed, base, "INFO", tuning.output.directory / "tuning.log").run()

    counts = count_trials(open_study(tuning))
    assert counts["complete"] + counts["pruned"] == 5


def test_runner_with_parallel_worker_processes(tuning_dict, caplog):
    caplog.set_level(logging.INFO)
    tuning_dict["budget"].update({"parallel_workers": 2, "max_trials": 4})
    tuning_dict["final_training"] = {"enabled": False}
    tuning, base = load(tuning_dict)

    TuningRunner(tuning, base, "INFO", tuning.output.directory / "tuning.log").run()

    counts = count_trials(open_study(tuning))
    assert counts["complete"] + counts["pruned"] >= 4
    assert "Workers     2 running" in caplog.text
    log = (tuning.output.directory / "tuning.log").read_text(encoding="utf-8")
    assert "[worker 0]" in log and "[worker 1]" in log


def test_cli_runs_tuning(tmp_path, tuning_dict):
    tuning_dict["final_training"] = {"enabled": False}

    assert main(["--config", str(write_tuning(tmp_path, tuning_dict))]) == ExitCode.SUCCESS
    assert (Path(tuning_dict["output"]["directory"]) / "tuning.log").is_file()


def test_cli_reports_preflight_failure_without_starting(tmp_path, tuning_dict, capsys):
    tuning_dict["search_space"]["latent_dim"] = {"low": 2, "high": 40}

    exit_code = main(["--config", str(write_tuning(tmp_path, tuning_dict))])

    assert exit_code == ExitCode.KNOWN_ERROR
    stderr = capsys.readouterr().err
    assert "TuningError" in stderr
    assert "ALL CHECKS PASSED" not in stderr


def load_data(base: ExperimentConfig):
    from fraud_detection.training.training_runner import TrainingRunner

    return TrainingRunner(base).load_datasets()
