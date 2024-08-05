import os
import streamlit as st
import torch
from transformers import BertForSequenceClassification
from src.data_preprocessing import Tokenizer

local_model_path = "models/bert"
intro = """
The following text box takes in any comment/statement and identifies the type of dysfunctional thought/belief it exhibits, if any.
These thought types draw from the Cognitive Behavioural Therapy (CBT) paradigm. They include: automatic thought, intermediate belief and core belief.
"""
intro
user_input = st.text_area("Insert a sentence:")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = os.path.join(os.getcwd(),local_model_path)
model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
tokenizer = Tokenizer("bert")
input_tokenized = tokenizer.tokenizer(user_input,max_length=200,padding='max_length',truncation=True)
with torch.no_grad():
    result = model(torch.LongTensor(input_tokenized["input_ids"]).unsqueeze(0).to(device))
pred_id = torch.argmax(result["logits"],dim=1).cpu().detach().numpy()[0]
labels_map = {0:"Automatic Thought",1:"Intermediate Belief",2:"Core Belief"}
if user_input:
    labels_map[pred_id]