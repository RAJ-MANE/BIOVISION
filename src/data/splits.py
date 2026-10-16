"""
Leakage-safe data splitting for produce tracking datasets.
Ensures identical fruit specimens or batches NEVER cross train/val/test splits.
"""
from typing import Dict, List, Tuple
import random
import numpy as np
import pandas as pd


def create_fruit_id_splits(
    metadata_df: pd.DataFrame,
    fruit_id_col: str = "fruit_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a metadata dataframe containing produce scans such that every individual
    fruit specimen (all its daily timepoints) appears EXCLUSIVELY in either train,
    val, or test.

    Args:
        metadata_df: DataFrame containing at least `fruit_id_col`
        fruit_id_col: column name grouping multiple scans of the same fruit
        train_ratio: fraction of unique fruits for training
        val_ratio: fraction of unique fruits for validation
        test_ratio: fraction of unique fruits for testing
        seed: random seed for reproducibility

    Returns:
        train_df, val_df, test_df
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-3), "Split ratios must sum to 1.0"
    
    unique_fruits = sorted(metadata_df[fruit_id_col].unique())
    rng = random.Random(seed)
    shuffled_fruits = list(unique_fruits)
    rng.shuffle(shuffled_fruits)

    n_total = len(shuffled_fruits)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    
    train_ids = set(shuffled_fruits[:n_train])
    val_ids = set(shuffled_fruits[n_train:n_train + n_val])
    test_ids = set(shuffled_fruits[n_train + n_val:])

    # Edge case: ensure test has at least 1 fruit if possible
    if len(test_ids) == 0 and len(train_ids) > 2:
        test_ids.add(shuffled_fruits[0])
        train_ids.remove(shuffled_fruits[0])

    train_df = metadata_df[metadata_df[fruit_id_col].isin(train_ids)].copy().reset_index(drop=True)
    val_df = metadata_df[metadata_df[fruit_id_col].isin(val_ids)].copy().reset_index(drop=True)
    test_df = metadata_df[metadata_df[fruit_id_col].isin(test_ids)].copy().reset_index(drop=True)

    # Verification: assert 0 intersection
    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage detected between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data leakage detected between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage detected between val and test!"

    return train_df, val_df, test_df
