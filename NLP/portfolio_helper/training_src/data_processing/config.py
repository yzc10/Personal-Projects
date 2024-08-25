import os

from langchain_openai import ChatOpenAI

data_path = "data"
news_data_config = {
    "date_from":"2024-01-01",
    "date_to":"2024-07-30",
    "tickers":["NVDA","SMCI","WBA","LULU","MSFT","LLY","SPY","QQQ","VTI","IWM"]
}