from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

load_dotenv()
with open('resume.txt','r') as file:
    resume = file.read()
model_config = {
    'temperature':1.2
}
model = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=model_config['temperature']
)
your_name = "Placeholder"
