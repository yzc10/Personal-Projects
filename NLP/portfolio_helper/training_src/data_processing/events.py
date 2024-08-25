from datetime import datetime
import json
import os
import requests
from tqdm.auto import tqdm
from typing import List,Dict

import pandas as pd

from training_src.data_processing.config import data_path,news_data_config

raw_data_path = os.path.join(data_path,"raw_data")

def get_news(ticker:str):
    date_from = news_data_config["date_from"]
    date_to = news_data_config["date_to"]
    limit = 500
    while datetime.strptime(date_from,"%Y-%m-%d") < datetime.strptime(date_to,"%Y-%m-%d"):
        news_file_path = os.path.join(raw_data_path,f"{ticker}_news_{date_to}.json")
        eod_api_url = f"https://eodhd.com/api/news?s={ticker}&from={date_from}&to={date_to}&limit={limit}&api_token=66b650c82847b1.48897437&fmt=json"
        response = requests.get(eod_api_url)
        temp_json = response.json()
        with open(news_file_path,"w") as f:
            json.dump(response.json(),f)
        prev_date_to = date_to
        date_idx = temp_json[-1]["date"].index("T")
        date_to = temp_json[-1]["date"][:date_idx]
        if (date_to == date_from) or (len(temp_json)<limit and prev_date_to==date_to): # why do I need this?
            break

# Consolidate news json into csv files
processed_data_path = os.path.join(data_path,"processed_data")

def get_sentiment_col(sentiment_val:Dict|None,col:str):
    if not pd.isnull(sentiment_val):
        return sentiment_val[col]
    else:
        return 0

def convert_json_to_csv(ticker_list:List[str]):
    if not os.path.isdir(processed_data_path):
        os.mkdir(processed_data_path)
    data_files_list = os.listdir(raw_data_path)
    news_files_list = [f for f in data_files_list if "news" in f and f[-4:]=="json"]
    news_files_dict = {}
    sample_file = news_files_list[0]
    with open(os.path.join(raw_data_path,sample_file),"r") as f:
        sample_json = json.load(f)
    df_cols = list(sample_json[0].keys())
    del sample_json
    for stock in tqdm(ticker_list):
        stock_df = pd.DataFrame(columns=df_cols)
        news_files_dict[stock] = [f for f in news_files_list if stock in f]
        for file in news_files_dict[stock]:
            with open(os.path.join(raw_data_path,file),"r") as f:
                temp_df = pd.DataFrame(json.load(f))
            stock_df = pd.concat([stock_df,temp_df])
        for col in ["polarity","neg","neu","pos"]:
            stock_df[f"sentiment_{col}"] = stock_df["sentiment"].apply(lambda x: get_sentiment_col(x,col))
        stock_df.drop(["sentiment","symbols","tags"],axis=1,inplace=True)
        stock_df["date"] = pd.to_datetime(stock_df["date"],format="%Y-%m-%dT%H:%M:%S%z")
        stock_df.sort_values(by=["date"],ascending=False,inplace=True)
        stock_df.reset_index(drop=True,inplace=True)
        stock_df.drop_duplicates(inplace=True)
        article_content = stock_df[["content"]].to_dict("index")
        with open(os.path.join(processed_data_path,f"{stock}_news_content.json"),"w") as f:
            json.dump(article_content,f)
        stock_df.drop(["content"],axis=1,inplace=True)
        stock_df.to_csv(os.path.join(processed_data_path,f"{stock}_news.csv"),index=True)


"""
python training_src\data_processing\events.py

Comments/Open questions:
- Too much hallucination in approach 1
- (Bonus) Developing k-means model to derive event categories (e.g. event releases, etc.)
"""