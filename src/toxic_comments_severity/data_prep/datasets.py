"""Module to obtain the datasets required for this project.
"""

import os
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split


class ToxicCommentsDataset:
    """
    Class to obtain the Toxic Comments dataset and clean the data for modeling.
    """

    def __init__(self, data_dir: str):
        """
        Initialize the dataset with the data directory.

        Args:
            data_dir: The directory where the dataset is stored

        Returns:
            None
        """
        self.data_dir = data_dir
        self.random_state = 42

    def load_polar(self) -> pl.DataFrame:
        """
        Load the dataset and return a polars dataframe.

        Returns:
            pl.DataFrame: polars dataframe containing the raw data
        """
        data = pl.read_csv(self.data_dir, has_header=True, ignore_errors=True)
        return data

    def load_pd(self) -> pd.DataFrame:
        """
        Load the dataset and return a pandas dataframe.

        Returns:
            pd.DataFrame: pandas dataframe containing the raw data
        """
        data = pd.read_csv(self.data_dir, low_memory=False)
        return data

    def _data_clean_polar(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset and return a cleaned polars dataframe.

        Args:
            data (pl.DataFrame): raw polars dataframe

        Returns:
            pl.DataFrame: cleaned polars dataframe with target columns and 7 toxic categories
        """

        # TODO: Add data cleaning steps here - in polars

        return None

    def _split_dataset(self, dataset, test_size=0.1, random_state=42):
        """
        Split the dataset into training and testing sets.

        Parameters:
            dataset (pandas.DataFrame): The dataset to be split.
            test_size (float): The proportion of the dataset to include in the test set.
            random_state (int): The seed used by the random number generator.

        Returns:
            tuple: A tuple containing the training set and the test set.
        """
        train_set, test_set = train_test_split(
            dataset, test_size=test_size, random_state=random_state
        )
        return train_set, test_set

    def save_datasets(self, train_set, test_set, train_path, test_path):
        """
        Save the train and test datasets to the specified file paths.

        Args:
            train_set (pandas.DataFrame): The train dataset.
            test_set (pandas.DataFrame): The test dataset.
            train_path (str): The file path to save the train dataset.
            test_path (str): The file path to save the test dataset.
        """
        train_set.to_pickle(train_path)
        test_set.to_pickle(test_path)

    def clean_data(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Cleans the provided data by performing the following steps:
        1. Fills missing values in the 'lang' column with 'en'.
        2. Filters the data to include only English comments.
        3. Selects specific columns of interest.
        4. Drops rows with missing values.
        5. Calculates the maximum toxicity score across different categories.
        6. Drops the individual toxicity columns.
        7. Randomly samples 25% of the data.
        8. Splits the data into train and test datasets.

        Args:
            data (pd.DataFrame, optional): The input data to be cleaned. If not provided, the default data will be used.

        Returns:
            pd.DataFrame: The cleaned train and test datasets.
        """
        data.lang = data.lang.fillna("en")
        data = data[data.lang == "en"]

        cols = [
            "comment_text",
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
            "sexual_explicit",
        ]
        data = data[cols]
        data = data.dropna()

        data["toxicity_score"] = data[cols[1:]].max(axis=1)
        data.drop(cols[1:], axis=1, inplace=True)
        data = data.sample(frac=0.25, random_state=self.random_state)
        train_data, test_data = self._split_dataset(data)

        return train_data, test_data
