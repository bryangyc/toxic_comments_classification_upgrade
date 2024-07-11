"""
This script is for training a model on the toxic comments dataset.
"""

import logging
import os

import dill as pickle
import hydra
import omegaconf
import torch
from huggingface_hub import login

import toxic_comments_severity as toxic_comments_severity


# pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="../conf/base", config_name="pipelines.yaml")
def main(args):
    """This is the main function for training the model.

    Parameters
    ----------
    args : omegaconf.DictConfig
        An omegaconf.DictConfig object containing arguments for the main function.
    """
    args = args["train_model"]

    # logger = logging.getLogger(__name__)
    # logger.info("Setting up logging configuration.")
    # toxic_comments_severity.general_utils.setup_logging(
    #     logging_config_path=os.path.join(
    #         hydra.utils.get_original_cwd(), "conf/base/logging.yaml"
    #     )
    # )

    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif not args["no_mps"] and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    with open(args["train_dataset"], "rb") as f:
        train_dataset = pickle.load(f)
    f.close()
    with open(args["test_dataset"], "rb") as f:
        test_dataset = pickle.load(f)
    f.close()

    if args["model_type"] == "fine_tuned":
        model = toxic_comments_severity.modeling.models.ToxicModelingFineTuned()
        pre_trained_model = (
            toxic_comments_severity.modeling.models.ToxicModelingFineTuned()
        )
        # test_sample = test_dataset.sample(20)
        pre_trained_model.pred_dataset(dataset=test_dataset, text_col=args["text_col"])
        pre_trained_model.save_dataset(dataset=test_dataset, name="test")
        rmse = pre_trained_model.evaluate(dataset=test_dataset)
        print(rmse)
    else:
        print("using base model")
        login()

        with open(args["train_dataset"], "rb") as f:
            train_dataset = pickle.load(f)

        with open(args["test_dataset"], "rb") as f:
            test_dataset = pickle.load(f)

        tx = toxic_comments_severity.modeling.models.ToxicModeling()

        # convert dataset from pandas to huggingface dataset
        # sampling to reduce training load
        train_dataset, test_dataset = tx.parse_pandas_to_dataset(
            train_dataset, test_dataset, sampling=args.sample_size
        )

        encoded_train_dataset = train_dataset.map(tx.tokenize_function, batched=True)
        encoded_test_dataset = test_dataset.map(tx.tokenize_function, batched=True)

        tx.train(
            encoded_train_dataset=encoded_train_dataset,
            encoded_eval_dataset=encoded_test_dataset,
            text_col=args["text_col"],
        )


if __name__ == "__main__":
    main()
