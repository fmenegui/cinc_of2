import pandas as pd
from sklearn.model_selection import GroupKFold

def group_k_fold(dataframe, group_column, n_splits=5):
    """
    Generate indices to split data into training and validation sets using
    Group K-Fold cross-validator.

    Parameters:
    dataframe (pd.DataFrame): The dataset containing group information.
    group_column (str): Column name that represents the group (e.g., patient_id).
    n_splits (int): Number of folds.

    Yields:
    dict: A dictionary with 'train' and 'valid' keys for each fold.
    """
    dataframe = dataframe.reset_index(drop=True)
    groups = dataframe[group_column]

    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, valid_idx in gkf.split(dataframe, groups=groups):
        yield {"train": train_idx, "valid": valid_idx, "test": None}


def group_k_fold_with_test(dataframe, group_column, n_splits=5):
    """
    Generate indices to split data into train, validation, and test sets for each fold using
    Group K-Fold cross-validator.

    Parameters:
    dataframe (pd.DataFrame): The dataset containing group information.
    group_column (str): Column name that represents the group (e.g., patient_id).
    n_splits (int): Number of folds.

    Yields:
    dict: A dictionary with 'train', 'valid', and 'test' keys for each fold.
    """
    dataframe = dataframe.reset_index(drop=True)
    groups = dataframe[group_column]

    gkf = GroupKFold(n_splits=n_splits)

    for train_valid_idx, test_idx in gkf.split(dataframe, groups=groups):
        # Splitting train_valid set into train and valid sets
        train_valid_set = dataframe.iloc[train_valid_idx]
        train_valid_groups = train_valid_set[group_column]
        gkf_train_valid = GroupKFold(n_splits=5)
        train_idx, valid_idx = next(gkf_train_valid.split(train_valid_set, groups=train_valid_groups))

        yield {
            "train": train_valid_set.iloc[train_idx].index,
            "valid": train_valid_set.iloc[valid_idx].index,
            "test": dataframe.iloc[test_idx].index,
        }
