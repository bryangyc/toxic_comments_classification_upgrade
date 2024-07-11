# Toxic comments severity

### Assessing Toxicity of Online comments


## Requirements

- **Python** == 3.11
- [PDM](https://pdm-project.org/latest/) - This project uses PDM to manages all dependencies. Please refer to [PDM documentation](https://pdm-project.org/latest/) to install accordingly.

## Pipeline Overview

![pipeline image](docs/pipeline.drawio.png)


## Dataset from Kaggle 

https://www.kaggle.com/datasets/shivamb/combined-jigsaw-comments-corpus

- This project predict the toxic severity of a given input, trained on a dataset of 2 million movie comments.  

- Goal is to predict score of toxic_comments_severity from 0 (a little toxic) to 1 (toxic)

- Gradio will be used to showcase the model, by  keying in any sentence, a predictation of the severity of toxicity will be shown.

- Model: https://huggingface.co/martin-ha/toxic-comment-model

## Demo with Gradio

To launch Gradio demo app, please ensure that all prerequisite are met.

Running app within pdm venv `python ./src/app.py`
