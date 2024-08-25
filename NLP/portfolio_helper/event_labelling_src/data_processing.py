"""
Run this after the event data for each stock has been extracted and consolidated into .csv files
"""
import json
import os
from tqdm.auto import tqdm
from typing import List,Dict

from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import RobertaTokenizer

from event_labelling_src.config import model
from event_labelling_src.prompts import GET_EVENT_LABELS_PROMPT

load_dotenv()
processed_data_path = "data/processed_data"
labelled_news_data_path = "data/labelled_news_data"
if not os.path.isdir(labelled_news_data_path):
    os.makedirs(labelled_news_data_path)

target_cols = ["date","article_index","title","stock"]
roberta_checkpoint = "FacebookAI/roberta-base"

# Send first batch of events to LLM for mass labelling
def compile_events(output_filename:str) -> None:
    """
    Combine stock-level event csv files into 1 master csv file.
    """
    news_files = [f for f in os.listdir(processed_data_path) if "news.csv" in f and f[:3].isupper()]
    all_df = pd.DataFrame(columns=target_cols)
    for file in tqdm(news_files):
        stock_df = pd.read_csv(
            os.path.join(processed_data_path,file),
            parse_dates=["date"],
            date_format="%Y-%m-%d %H:%M:%S%z"
        )
        stock_df["day"] = stock_df["date"].dt.date
        stock_df["hour"] = stock_df["date"].dt.hour
        stock_df2 = stock_df.groupby(["hour","day"]).first()
        stock_df2.rename(columns={"Unnamed: 0":"article_index"},inplace=True)
        stock_df2["stock"] = file.split("_")[0]
        stock_df2 = stock_df2[~(stock_df2["link"].str.contains("fool.com")|stock_df2["link"].str.contains("investorplace.com"))]
        stock_df2.sort_values(by="date",ascending=False,inplace=True)
        stock_df2["title"] = stock_df2["title"].apply(lambda x:x.replace("\"","'"))
        all_df = pd.concat([all_df,stock_df2[target_cols]])
    all_df.to_csv(os.path.join(processed_data_path,output_filename),index=False)

## Send titles to LLM, get labels
class LabeledTitle(BaseModel):
    title: str = Field(description="title of article")
    stock: str = Field(description="ticker symbol of stock to which the article pertains. It is the string value in the first parentheses, where all letters are in upper case. e.g.MSFT.")
    article_idx: str = Field(description="index of article. It is the integer value in the second parentheses, e.g.8")
    is_news: str = Field(description="whether the article refers to a piece of news or not. Its value should be yes or no")

class TitlesList(BaseModel):
    content: List[LabeledTitle] = Field(description="list of labelled article titles")

def get_labels(news_data_filename:str):
    """
    Get event labels for selected events from master .csv file. Save results in batched json files.
    """
    article_df = pd.read_csv(os.path.join(processed_data_path,news_data_filename),index_col=False)
    total_num = len(article_df)
    batch_size = 50
    batch_num = total_num//batch_size
    parser = JsonOutputParser(pydantic_object=TitlesList)
    prompt = PromptTemplate(
        template=GET_EVENT_LABELS_PROMPT,
        input_variables=["articles_list"],
        partial_variables={"format_instructions":parser.get_format_instructions()}
    )
    if total_num%batch_size != 0:
        batch_num += 1
    batches = [article_df.iloc[i*batch_size:(i+1)*batch_size].reset_index() for i in range(batch_num)]
    for i in tqdm(range(batch_num)):
        article_str = ""
        for j in range(len(batches[i])):
            article_str += "- " + batches[i].at[j,"title"] + f"({batches[i].at[j,'stock']})({batches[i].at[j,'article_index']})" + "\n"
        chain = prompt | model | parser
        reply = chain.invoke({"articles_list":article_str})
        json_filename = f"batch_{i}_news.json"
        with open(os.path.join(labelled_news_data_path,json_filename),"w") as f:
            json.dump(reply,f)

def compile_labelled_data(output_filename:str) -> None:
    """
    Consolidate batched json files with event labels into updated master csv file.
    """
    labelled_news_batches = os.listdir(labelled_news_data_path)
    labelled_df = pd.read_csv(
        os.path.join(processed_data_path,"all_news_v0.csv"),
        parse_dates=["date"],
        date_format="%Y-%m-%d %H:%M:%S%z"
        )
    labelled_df_cols = ["stock","article_index","is_news"]
    temp_df = pd.DataFrame(columns=labelled_df_cols)
    for file in tqdm(labelled_news_batches):
        with open(os.path.join(labelled_news_data_path,file)) as f:
            batch_df = pd.DataFrame(json.load(f))
        batch_df.rename(columns={"article_idx":"article_index"},inplace=True)
        batch_df = batch_df[labelled_df_cols]
        temp_df = pd.concat([temp_df,batch_df])
    labelled_df = pd.merge(
        left=labelled_df,
        right=temp_df,
        how="left",
        left_on=["stock","article_index"],
        right_on=["stock","article_index"]
        )
    labelled_df.to_csv(os.path.join(processed_data_path,output_filename),index=False)

def convert_target(value):
    if value == "yes": return 1
    else: return 0

class ProcessedDataset:
    def __init__(self,events_filepath:str):
        self.df = pd.read_csv(
            events_filepath,
            parse_dates=["date"],
            date_format="%Y-%m-%d %H:%M:%S%z"
            )
        self.df["label"] = self.df["is_news"].map(convert_target)
        self.df = self.df[["title","label"]]
        self.tokenizer = RobertaTokenizer.from_pretrained(
            roberta_checkpoint,
            do_lower_case=True
        )
    
    def _preprocess(self,sample:Dict)->Dict:
        text = sample["title"]
        encoding = self.tokenizer(
            text,
            max_length=100,
            padding="max_length",
            truncation=True
        )
        # labels_matrix = np.zeros(len(text)).astype(int)
        # for i in range(len(text)):
        #     if sample["label"][i] == 1:
        #         labels_matrix[i] = 1
        # encoding["labels"] = torch.from_numpy(labels_matrix)
        encoding["labels"] = torch.from_numpy(np.array(sample["label"]).astype(int))
        return encoding

    def generate_datasets(self):
        self.df_core, self.df_test = train_test_split(
            self.df,
            test_size=0.2,
            random_state=42
            )
        self.df_core.reset_index(drop=True,inplace=True)
        self.df_test.reset_index(drop=True,inplace=True)
        self.ds_core = Dataset.from_pandas(self.df_core)
        self.ds_core = self.ds_core.map(self._preprocess,batched=True)
        self.ds_core.set_format("torch",columns=['input_ids', 'attention_mask', 'labels'])
        self.ds_test = Dataset.from_pandas(self.df_test)
        self.ds_test = self.ds_test.map(self._preprocess,batched=True)
        self.ds_test.set_format("torch",columns=['input_ids', 'attention_mask', 'labels'])

