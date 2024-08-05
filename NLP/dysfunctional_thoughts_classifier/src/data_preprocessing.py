import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizerFast, RobertaTokenizer, DistilBertTokenizer

models = {'bert':'bert-base-uncased',
          'roberta':'FacebookAI/roberta-base',
          'distilbert':'distilbert-base-uncased'
}
labels = ['Automatic Thought','Intermediate Belief','Core Belief']

def convert_target(value):
    """
    """
    if value == 'Automatic Thought': return 0
    elif value == 'Intermediate Belief': return 1
    else: return 2

class Tokenizer:
    def __init__(self,model_type):
        if model_type == 'bert':
            self.checkpoint = models['bert']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.tokenizer = BertTokenizerFast.from_pretrained(self.checkpoint,do_lower_case=True) # Check
        elif model_type == 'distilbert':
            self.checkpoint = models['distilbert']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.tokenizer = DistilBertTokenizer.from_pretrained(self.checkpoint,do_lower_case=True) # Check  
        elif model_type == 'roberta':
            self.checkpoint = models['roberta']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.tokenizer = RobertaTokenizer.from_pretrained(self.checkpoint,do_lower_case=True) # Check  
        else: 
            raise Exception("Input a valid model type: bert|distilbert|roberta")


class Data_preprocessed:
    def __init__(self,df_path):
        self.df = pd.read_csv(df_path)
    
    def preprocess(self,prev_turns=False):
        self.df['label'] = self.df['Type_of_Thought/Belief'].map(convert_target)
        if prev_turns == False:
            self.df = self.df.drop(columns='Previous_Turns',axis=1)
            self.df = self.df.rename(columns={'Current_Turn':'text'})
        else:
            self.df['text'] = self.df['Previous_Turns'] + ' ' + self.df['Current_Turn']
            self.df = self.df.drop(columns=['Previous_Turns','Current_Turn'],axis=1)
        self.df = self.df.drop(columns=['article_number','Type_of_Thought/Belief'],axis=1) 
        self.columns = self.df.columns
    
    def generate_datasets(self,model_type):
        self.tokenizer = Tokenizer(model_type)
        self.df1, self.df_test = train_test_split(self.df,test_size=0.2,random_state=42)
        self.df1, self.df_test = self.df1.reset_index(drop=True), self.df_test.reset_index(drop=True)
        def preprocess(sample):
            text = sample['text']
            encoding = self.tokenizer.tokenizer(text,max_length=200,padding='max_length',truncation=True) #self.tokenizer.tokenizer
            labels_matrix = np.zeros((len(text),len(labels)))
            for i in range(len(text)):
                for j in range(len(labels)):
                    if sample['label'][i] == j:
                        labels_matrix[i,j] = 1 
            encoding['labels'] = labels_matrix.tolist()
            return encoding
        self.core_ds = Dataset.from_pandas(self.df)
        self.core_ds = self.core_ds.map(preprocess,batched=True)
        self.core_ds.set_format('torch')
        self.test_ds = Dataset.from_pandas(self.df)
        self.test_ds = self.test_ds.map(preprocess,batched=True)
        self.test_ds.set_format('torch')