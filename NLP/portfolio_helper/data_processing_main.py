import argparse
import os
from tqdm.auto import tqdm

from training_src.data_processing.config import news_data_config
from training_src.data_processing.events import get_news,convert_json_to_csv
from event_labelling_src.data_processing import compile_events,get_labels,compile_labelled_data
from event_labelling_src.model import ModelTrainer

if __name__ == "__main__":
    # Collect events data (individual stock data has already been collected via EDA)
    for stock in tqdm(news_data_config["tickers"]):
        get_news(ticker=stock)
    convert_json_to_csv(news_data_config["tickers"])

    # Get event labels (is the event a piece of news or not)
    compile_events("all_news_v0.csv")
    get_labels("all_news_v0.csv")
    compile_labelled_data("all_news_labelled_v0.csv")

    # Train & evaluate event-labelling model
    model_dir = "models/roberta"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    trainer = ModelTrainer(model_dir)
    trainer.train()
    trainer.evaluate()