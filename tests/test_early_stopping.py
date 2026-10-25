from fraud_detection.training.early_stopping import EarlyStopping


def test_early_stopping_triggers_after_patience():
    stopping = EarlyStopping(patience=2, min_delta=0.01)

    assert stopping.update(1.0)
    assert not stopping.update(0.995)
    assert not stopping.should_stop
    assert not stopping.update(1.2)
    assert stopping.should_stop
    assert stopping.best_loss == 1.0


def test_improvement_resets_patience():
    stopping = EarlyStopping(patience=2)
    stopping.update(1.0)
    stopping.update(1.1)
    stopping.update(0.5)
    stopping.update(0.6)

    assert not stopping.should_stop


def test_disabled_early_stopping_never_stops():
    stopping = EarlyStopping(patience=1, enabled=False)
    stopping.update(1.0)
    stopping.update(2.0)

    assert not stopping.should_stop
