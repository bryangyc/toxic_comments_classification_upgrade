import random
import time

import gradio as gr
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

MODEL_HUGGINGFACE_PATH = "martin-ha/toxic-comment-model"
EXAMPLE = [
    "Free trade has been the 'key' to our prosperity for a quarter century.",
    "See above reply to your dumb***.",
]


model_path = MODEL_HUGGINGFACE_PATH
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)


def text_classification(text: str) -> str:
    """Returns a score to rank the toxicity of the sentence.

    Args:
        text (str): a sentence input.

    Returns:
        str: f-string that outputs the score.
    """
    result = classifier(text)
    print(result)
    is_toxic = False

    sentiment_label = result[0]["label"]

    if sentiment_label == "non-toxic":
        sentiment_score = 1 - result[0]["score"]
    else:
        sentiment_score = result[0]["score"]
    if sentiment_score > 0.2:
        is_toxic = True
    time.sleep(0.5)

    # return f"Toxicity Level: {round(sentiment_score, 3)}"
    return "More toxic" if is_toxic else "Not so toxic"


def update_slider(text: str) -> float:
    """Returns the score in float format to update slider value

    Args:
        text (str): a sentence input.

    Returns:
        float: a float score rounded to 3 decimal.
    """
    result = classifier(text)
    sentiment_label = result[0]["label"]
    if sentiment_label == "non-toxic":
        sentiment_score = 1 - result[0]["score"]
    else:
        sentiment_score = result[0]["score"]
    time.sleep(0.5)
    return round(sentiment_score, 3)


def get_toxicity_level(score: float) -> str:
    """Returning Toxicity level between 0 to 1

    Args:
        score (float): Toxic score from model output

    Returns:
        str: Description from chatbot
    """
    not_so_toxic = f"This statement is not so toxic, score {score} of 1"
    toxic = "I am going to report you."

    if score < 0.2:
        return not_so_toxic
    return toxic


def respond(message: str, chat_history: gr.Chatbot) -> tuple[str, gr.Chatbot]:
    """_summary_

    Args:
        message (str): _description_
        chat_history (gr.Chatbot): _description_

    Returns:
        tuple[str, gr.Chatbot]: _description_
    """
    model_score = update_slider(message)

    bot_message = random.choice(["Indeed", "I think otherwise", "That is so true!"])
    bot_message = get_toxicity_level(model_score)
    chat_history.append((message, bot_message))
    time.sleep(0.5)
    print(model_score)
    return "", chat_history


# main block
with gr.Blocks(theme="snehilsanyal/scikit-learn") as demo:
    gr.Markdown("# Toxicity Detector)")
    # row 1
    with gr.Row():
        chatbot = gr.Chatbot()
    # row 2
    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Send your message", lines=1, placeholder="Say something!"
            )
            send_btn = gr.Button("Send Message")
        with gr.Column(scale=1):
            result_text = gr.Textbox(label="Toxicity Label", placeholder="Toxicity")
            slider = gr.Slider(0, 1, value=0, label="Proba", min_width=0)
    # row 3
    with gr.Row():
        exp = gr.Examples(EXAMPLE, input_text)

    # actions idenfified
    # button click
    # text input on value change
    # text submit with shift+enter submit

    # actions to update result txt
    send_btn.click(text_classification, input_text, result_text)
    input_text.input(text_classification, input_text, result_text)
    input_text.submit(text_classification, input_text, result_text)

    # action to update chat bot updates
    send_btn.click(respond, [input_text, chatbot], [input_text, chatbot])
    input_text.submit(respond, [input_text, chatbot], [input_text, chatbot])

    # action to update slider with similar proba
    send_btn.click(update_slider, input_text, slider)
    input_text.input(update_slider, input_text, slider)
    input_text.submit(update_slider, input_text, slider)


if __name__ == "__main__":
    demo.launch(share=True)
