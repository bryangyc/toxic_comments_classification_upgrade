"""This script processes raw data and saves them in the processed data directory.
"""

import os
import shutil
import logging
import pandas as pd

import hydra

import toxic_comments_severity as toxic_comments_severity


# pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="../conf/base", config_name="pipelines.yaml")
def main(args):
    """This function processes raw data and saves them in the processed data directory.

    Parameters
    ----------
    args : omegaconf.DictConfig
        An omegaconf.DictConfig object containing arguments for the main function.
    """
    args = args["process_data"]

    raw_data_dir_path = args["raw_data_dir_path"]
    raw_data_file_path = os.path.join(raw_data_dir_path, "combined_jigsaw_comments.csv")

    processed_data_dir_path = args["processed_data_dir_path"]
    os.makedirs(args["processed_data_dir_path"], exist_ok=True)

    process = toxic_comments_severity.data_prep.datasets.ToxicCommentsDataset(
        data_dir=raw_data_file_path
    )

    raw_df = process.load_pd()
    processed_df_train, processed_df_test = process.clean_data(raw_df)

    train_save_path = os.path.join(processed_data_dir_path, "train.pkl")
    test_save_path = os.path.join(processed_data_dir_path, "test.pkl")
    process.save_datasets(
        train_set=processed_df_train,
        test_set=processed_df_test,
        train_path=train_save_path,
        test_path=test_save_path,
    )

    print("All raw data has been processed. And saved in the processed data directory.")


if __name__ == "__main__":
    main()
