"""Module for containing architectures/definition of models.
"""

import os

import dill as pickle
import evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from torch import cuda
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TextClassificationPipeline,
    Trainer,
    TrainingArguments,
)


class ToxicModelingFineTuned:
    """
    A class representing a fine-tuned toxic comments severity model.

    Attributes:
        model_path (str): The path to the pre-trained model.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the input text.
        model (AutoModelForSequenceClassification): The fine-tuned model for sequence classification.
        device (str): The device used for inference (e.g., "cuda" or "cpu").

    Methods:
        _predict(text): Performs prediction on a single text.
        save_dataset(dataset, name): Saves the dataset to a pickle file.
        evaluate(dataset): Evaluates the model's performance on a dataset.
        pred_dataset(dataset, text_col): Performs prediction on a dataset.

    """

    def __init__(self):
        self.model_path = "martin-ha/toxic-comment-model"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.device = "cuda" if cuda.is_available() else "cpu"

    def _predict(self, text: str) -> object:
        """
        Predicts the severity of toxic comments based on the given text.

        Args:
            text (str): The input text to predict the severity of.

        Returns:
            object: The pipeline object containing the predicted label and score.
        """
        pipeline = TextClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer
        )
        return pipeline(text)

    def save_dataset(self, dataset: pd.DataFrame, name=None) -> None:
        """
        Saves the dataset to a pickle file.

        Args:
            dataset (pandas.DataFrame): The dataset to save.
            name (str, optional): The name of the saved file. Defaults to None.

        """
        if not os.path.exists("./data/pred"):
            os.makedirs("./data/pred")
        with open(f"./data/pred/{name}_fine_tuned_pre_trained_model.pkl", "wb") as f:
            pickle.dump(dataset, f)

    def evaluate(self, dataset: pd.DataFrame) -> float:
        """
        Evaluates the model's performance on a dataset.

        Args:
            dataset (pandas.DataFrame): The dataset to evaluate.

        Returns:
            float: The root mean squared error (RMSE) of the model's predictions.

        """
        mse = mean_squared_error(
            dataset["toxicity_score"], dataset["predicted_toxicity_score"]
        )
        return mse**0.5

    def pred_dataset(self, dataset: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Performs prediction on a dataset.

        Args:
            dataset (pandas.DataFrame): The dataset to predict.
            text_col (str): The name of the column containing the text data.

        Returns:
            pandas.DataFrame: The dataset with predicted labels and scores.
        """
        for index, row in dataset.iterrows():
            x = self._predict(row[text_col])
            dataset.at[index, "label"] = x[0]["label"]
            dataset.at[index, "score"] = x[0]["score"]
            if x[0]["label"] == "toxic":
                dataset.at[index, "predicted_toxicity_score"] = x[0]["score"]
            else:
                dataset.at[index, "predicted_toxicity_score"] = 1 - x[0]["score"]

        return dataset


## If doing fine-tuning
class ToxicModeling:

    def __init__(self):
        self.model_path = "distilbert/distilbert-base-uncased"
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        os.makedirs("./models/checkpoint", exist_ok=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=1
        ).to(self.device)
        self._training_args = self.training_args

    @property
    def training_args(self):
        """training arguments for Trainer module"""
        return TrainingArguments(
            output_dir="./models/base_toxicty",
            num_train_epochs=2,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=True,
            report_to="none",
        )

    def tokenize_function(self, example):
        return self.tokenizer(
            example["comment_text"], padding="max_length", truncation=True
        )

    def _compute_metrics(self, eval_pred):
        """Metrics for regression using SK learn functions"""
        logits, labels = eval_pred
        labels = labels.reshape(-1, 1)

        mse = mean_squared_error(labels, logits)

        # mean_sqaure_error with squared false is depreciated, using root_mean_sq_error fn
        rmse = root_mean_squared_error(labels, logits)
        return {"mse": mse, "rmse": rmse}

    @staticmethod
    def get_score_label(data: pd.DataFrame) -> str:
        """Search df columns for the first match of 'score' columns"""
        for text in data.columns:
            if "score" in text:
                return text

    def parse_pandas_to_dataset(
        self, train_df: pd.DataFrame, eval_df: pd.DataFrame, sampling: int = None
    ) -> tuple[Dataset, Dataset]:

        score_label = self.get_score_label(train_df)
        train_df = train_df.rename(columns={score_label: "label"})
        eval_df = eval_df.rename(columns={score_label: "label"})

        # to get smaller dataset for training purpose
        if sampling:
            train_dataset = Dataset.from_pandas(
                train_df[:sampling], preserve_index=False
            )
            eval_dataset = Dataset.from_pandas(eval_df[:sampling], preserve_index=False)
        else:
            train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
            eval_dataset = Dataset.from_pandas(eval_df, preserve_index=False)

        return train_dataset, eval_dataset

    def train(
        self,
        encoded_train_dataset: Dataset,
        encoded_eval_dataset: Dataset,
        text_col: str,
    ):

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=encoded_train_dataset,
            eval_dataset=encoded_eval_dataset,
            compute_metrics=self._compute_metrics,
            data_collator=data_collator,
        )
        trainer.train()
        return trainer
