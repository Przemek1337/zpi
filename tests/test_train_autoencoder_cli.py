from train_autoencoder import ExitCode, main


def test_successful_run_returns_success(experiment_dict, write_config):
    assert main(["--config", str(write_config(experiment_dict))]) == ExitCode.SUCCESS


def test_missing_config_returns_known_error(tmp_path, capsys):
    exit_code = main(["--config", str(tmp_path / "missing.yaml")])

    assert exit_code == ExitCode.KNOWN_ERROR
    assert "ConfigurationError" in capsys.readouterr().err


def test_data_error_is_logged_without_traceback(experiment_dict, write_config, tmp_path, capsys):
    experiment_dict["data"]["train_path"] = str(tmp_path / "absent.csv")

    exit_code = main(["--config", str(write_config(experiment_dict))])

    stderr = capsys.readouterr().err
    assert exit_code == ExitCode.KNOWN_ERROR
    assert "DataLoadingError" in stderr
    assert "Traceback" not in stderr


def test_debug_level_includes_traceback(experiment_dict, write_config, tmp_path, capsys):
    experiment_dict["data"]["train_path"] = str(tmp_path / "absent.csv")

    main(["--config", str(write_config(experiment_dict)), "--log-level", "DEBUG"])

    assert "Traceback" in capsys.readouterr().err
