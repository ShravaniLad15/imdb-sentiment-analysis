import gradio as gr
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

MODEL_PATH = "distilbert_imdb"

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_PATH
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    confidence = float(probs.max())

    prediction = (
        "Positive"
        if probs.argmax().item() == 1
        else "Negative"
    )

    return prediction, f"{confidence:.2%}"

demo = gr.Interface(
    fn=predict_sentiment,

    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter a movie review..."
    ),

    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Confidence")
    ],

    title="IMDB Sentiment Analysis",
    description="DistilBERT fine-tuned on IMDB movie reviews"
)

if __name__ == "__main__":
    demo.launch()
