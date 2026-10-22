import numpy as np
import pytest
from pydantic import ValidationError

from fraud_detection.data.transaction_set import TransactionSet
from tests.conftest import make_transaction_set


def test_label_filters_split_classes():
    transactions = make_transaction_set(legitimate=7, fraud=3)

    assert len(transactions.with_label(0)) == 7
    assert len(transactions.excluding_label(0)) == 3
    assert transactions.class_counts() == {0: 7, 1: 3}


def test_mismatched_row_counts_are_rejected():
    with pytest.raises(ValidationError, match="rows"):
        TransactionSet(features=np.zeros((3, 2)), labels=np.zeros(4), feature_names=("a", "b"))


def test_mismatched_feature_names_are_rejected():
    with pytest.raises(ValidationError, match="names"):
        TransactionSet(features=np.zeros((3, 2)), labels=np.zeros(3), feature_names=("a",))


def test_one_dimensional_features_are_rejected():
    with pytest.raises(ValidationError, match="2D"):
        TransactionSet(features=np.zeros(3), labels=np.zeros(3), feature_names=())
