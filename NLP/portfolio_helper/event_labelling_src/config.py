import os

from langchain_openai import ChatOpenAI

model_config = {
    "temperature":0.9
}
model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=model_config["temperature"]
)