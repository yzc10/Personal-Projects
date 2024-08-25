import os

import evaluate
import numpy as np
from sklearn.model_selection import KFold
import torch
from transformers import RobertaForSequenceClassification
from transformers import TrainingArguments, Trainer

from event_labelling_src.data_processing import ProcessedDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
train_params = {
    "lr":2e-15,
    'per_device_train_batch_size':8,
    'per_device_eval_batch_size':8,
    'num_train_epochs':2,
    'weight_decay':0.01,
}
eval_params = {
    "learning_rate":[2e-5,3e-5],
    "per_device_train_batch_size":[8,16]
}
roberta_checkpoint = "FacebookAI/roberta-base"
acc = evaluate.load("accuracy")
prec = evaluate.load("precision")
rec = evaluate.load("recall")
f1 = evaluate.load("f1")

def preprocess_logits_for_metrics(logits,labels): # Review
    pred_ids = torch.argmax(logits,dim=1)
    return pred_ids,labels

def compute_metrics(eval_pred):
    pred, labels = eval_pred
    if isinstance(pred,tuple): pred = pred[0] # for evaluation loops to run with preprocess_logits_for_metrics
    # labels = np.argmax(labels,axis=1) # Not needed as it isn't multi-class
    acc_score = [x for x in acc.compute(predictions=pred,references=labels).values()][0]
    prec_score = [x for x in prec.compute(predictions=pred,references=labels,average='macro').values()][0]
    rec_score = [x for x in rec.compute(predictions=pred,references=labels,average='macro').values()][0]
    f1_score = [x for x in f1.compute(predictions=pred,references=labels,average='macro').values()][0]
    return {'accuracy':acc_score,'precision':prec_score,'recall':rec_score,'f1':f1_score}

class ModelTrainer:
    def __init__(self,model_dir:str):
        self.dataset = ProcessedDataset("data/processed_data/all_news_labelled_v0.csv")
        self.dataset.generate_datasets()
        self.ds_core = self.dataset.ds_core.train_test_split(test_size=0.2,seed=42) # includes train & val sets
        self.ds_test = self.dataset.ds_test
        self.model_dir = model_dir
        self.train_args = TrainingArguments(
            output_dir = self.model_dir,
            learning_rate = train_params['lr'],
            per_device_train_batch_size = train_params['per_device_train_batch_size'],
            per_device_eval_batch_size = train_params['per_device_eval_batch_size'],
            num_train_epochs = train_params['num_train_epochs'],
            weight_decay = train_params['weight_decay'],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,
            report_to=['tensorboard']
        )
    
    def _train_with_cross_val(self):
        kfold = KFold(n_splits=3)
        splits = kfold.split(np.arange(self.dataset.ds_core.shape[0]))
        results = []
        best_loss = float("inf")
        for train_idx,val_idx in splits:
            self.model = RobertaForSequenceClassification.from_pretrained(
                roberta_checkpoint,
                problem_type="single_label_classification"
            )
            self.trainer = Trainer(
                model=self.model.to(device),
                args=self.train_args,
                train_dataset=self.dataset.ds_core.select(train_idx),
                eval_dataset=self.dataset.ds_core.select(val_idx),
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )
            self.trainer.args.dataloader_num_workers = 0 #Check
            self.trainer.args.dataloader_prefetch_factor = None #Check
            self.train_args.set_logging(report_to=['tensorboard']) #Check
            self.train_args.set_dataloader(train_batch_size=8,eval_batch_size=8) #Check
            self.trainer.train()
            val_result = self.trainer.evaluate()
            results.append(val_result)
            if val_result['eval_loss'] < best_loss:
                best_loss = val_result['eval_loss']
                self.chosen_model = self.model
    
    def _train_with_all_data(self):
        self.trainer = Trainer(
            model=self.chosen_model.to(device),
            args=self.train_args,
            train_dataset=self.ds_core["train"],
            eval_dataset=self.ds_core["test"],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.trainer.args.dataloader_num_workers = 0 #Check
        self.trainer.args.dataloader_prefetch_factor = None #Check
        self.train_args.set_logging(report_to=['tensorboard']) #Check
        self.train_args.set_dataloader(train_batch_size=8,eval_batch_size=8) #Check (previuosly 4,4)
        self.trainer.train()
        self.trainer.save_model(self.model_dir)
      
    def train(self):
        self._train_with_cross_val()
        self._train_with_all_data()
   
    def evaluate(self):
        model = RobertaForSequenceClassification.from_pretrained(self.model_dir).to(device)
        trainer = Trainer(
            model=model,
            args=self.train_args,
            train_dataset=self.ds_core["train"],
            eval_dataset=self.ds_core["test"],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        trainer.args.dataloader_num_workers = 0
        trainer.args.dataloader_prefetch_factor = None
        metrics = trainer.evaluate(self.data.test_ds)
        print(metrics)

