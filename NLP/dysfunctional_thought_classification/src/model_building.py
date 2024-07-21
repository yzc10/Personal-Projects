import numpy as np
from sklearn.model_selection import KFold
from src.data_preprocessing import *
import torch
from transformers import BertForSequenceClassification, DataCollatorWithPadding
from transformers import RobertaForSequenceClassification
from transformers import DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate
import os

os.environ['WANDB_DISABLED'] = 'true'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
params = {
    'lr': 2e-5,
    'per_device_train_batch_size':8,
    'per_device_eval_batch_size':8,
    'num_train_epochs':2,
    'weight_decay':0.01,
}
params_eval = {
    'learning_rate':[2e-5,3e-5],
    'per_device_train_batch_size':[8,16],
}
acc = evaluate.load('accuracy')
prec = evaluate.load('precision')
rec = evaluate.load('recall')
f1 = evaluate.load('f1')

def preprocess_logits_for_metrics(logits, labels):
    """
    Contingency workaround for OOM issue, due to potential memory leak in original Trainer class
    """
    pred_ids = torch.argmax(logits, dim=1)
    return pred_ids, labels

def compute_metrics(eval_pred):
    pred, labels = eval_pred
    if isinstance(pred,tuple): pred = pred[0] # for evaluation loops to run with preprocess_logits_for_metrics
    labels = np.argmax(labels,axis=1)
    acc_score = [x for x in acc.compute(predictions=pred,references=labels).values()][0]
    prec_score = [x for x in prec.compute(predictions=pred,references=labels,average='macro').values()][0]
    rec_score = [x for x in rec.compute(predictions=pred,references=labels,average='macro').values()][0]
    f1_score = [x for x in f1.compute(predictions=pred,references=labels,average='macro').values()][0]
    return {'accuracy':acc_score,'precision':prec_score,'recall':rec_score,'f1':f1_score}

class Text_CLF:
    def __init__(self,model_type):
        '''
        model_type (str): bert | roberta
        '''
        if model_type == 'bert':
            self.checkpoint = models['bert']
            self.model = BertForSequenceClassification.from_pretrained(
                self.checkpoint,
                problem_type='multi_label_classification',
                num_labels=3,
                id2label = {idx:label for idx,label in enumerate(labels)},
                label2id = {label:idx for idx,label in enumerate(labels)}
            )
        elif model_type == 'distilbert':
            self.checkpoint = models['distilbert']
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.checkpoint,
                problem_type='multi_label_classification',
                num_labels=3,
                id2label = {idx:label for idx,label in enumerate(labels)},
                label2id = {label:idx for idx,label in enumerate(labels)}
            ) 
        elif model_type == 'roberta':
            self.checkpoint = models['roberta']
            self.model = RobertaForSequenceClassification.from_pretrained(
                self.checkpoint,
                problem_type='multi_label_classification',
                num_labels=3,
                id2label = {idx:label for idx,label in enumerate(labels)},
                label2id = {label:idx for idx,label in enumerate(labels)}
            )            

class Model:
    def __init__(self,df_path:str,model_type='bert',prev_turns=False):
        self.data = Data_preprocessed(df_path)
        self.model_type = model_type
        self.data.preprocess(prev_turns)
        self.data.generate_datasets(self.model_type)
        self.core_ds = self.data.core_ds.train_test_split(test_size=0.2,seed=42)

    def train(self,model_dir:str,cv=True) -> None:
        # data_collator = DataCollatorWithPadding(tokenizer=self.mod.tokenizer)
        self.train_args = TrainingArguments(
            output_dir = model_dir,
            learning_rate = params['lr'],
            per_device_train_batch_size = params['per_device_train_batch_size'],
            per_device_eval_batch_size = params['per_device_eval_batch_size'],
            num_train_epochs = params['num_train_epochs'],
            weight_decay = params['weight_decay'],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,
            report_to=['tensorboard']
        )

        ### Cross Validation
        if cv == True:
            kfold = KFold(n_splits=3)
            splits = kfold.split(np.arange(self.data.core_ds.shape[0]))
            results = []
            best_loss = float('inf')
            for train_idx,val_idx in splits:    
                self.mod = Text_CLF(self.model_type)
                self.trainer = Trainer(
                    model=self.mod.model.to(device),
                    args=self.train_args,
                    train_dataset=self.data.core_ds.select(train_idx),
                    eval_dataset=self.data.core_ds.select(val_idx),
                    # data_collator= data_collator,
                    compute_metrics=compute_metrics,
                    preprocess_logits_for_metrics=preprocess_logits_for_metrics
                )
                self.trainer.args.dataloader_num_workers = 0
                self.trainer.args.dataloader_prefetch_factor = None
                self.train_args.set_logging(report_to=['tensorboard']) #TBC or ['tensorboard']
                self.train_args.set_dataloader(train_batch_size=4,eval_batch_size=4) #TBC
                self.trainer.train()
                val_result = self.trainer.evaluate()
                results.append(val_result)
                if val_result['eval_loss'] < best_loss:
                    best_loss = val_result['eval_loss']
                    self.chosen_model = self.mod.model
        else:
            self.mod = Text_CLF(self.model_type)
            self.chosen_model = self.mod.model
        
        ### Train best_model on all data
        self.trainer = Trainer(
            model=self.chosen_model.to(device),
            args=self.train_args,
            train_dataset=self.core_ds['train'],
            eval_dataset=self.core_ds['test'],
            # data_collator= data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.trainer.args.dataloader_num_workers = 0
        self.trainer.args.dataloader_prefetch_factor = None
        self.train_args.set_logging(report_to=['tensorboard'])
        self.train_args.set_dataloader(train_batch_size=4,eval_batch_size=4) #TBC
        self.trainer.train()
        self.trainer.save_model(model_dir)
    
    def evaluate(self,model_dir:str) -> None:
        if self.model_type == 'bert':
            model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
        elif self.model_type == 'distilbert':
            model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)            
        elif self.model_type == 'roberta':
            model = RobertaForSequenceClassification.from_pretrained(model_dir).to(device)            
        trainer = Trainer(
            model=model,
            args=self.train_args,
            train_dataset=self.core_ds['train'],
            eval_dataset=self.core_ds['test'],
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        trainer.args.dataloader_num_workers = 0
        trainer.args.dataloader_prefetch_factor = None
        metrics = trainer.evaluate(self.data.test_ds)
        print(metrics)

    def tune_param(self,model_dir:str):
        param_keys = list(params_eval.keys())
        param_combs = np.array(np.meshgrid(*[params_eval[k] for k in params_eval])).T.reshape(-1,len(param_keys))
        results = []
        best_loss = float('inf')
        for i in range(len(param_combs)):
            param_args = {}
            for k in range(len(param_keys)):
                if 'size' in param_keys[k]: 
                    param_args[param_keys[k]] = int(param_combs[i][k])
                else:
                    param_args[param_keys[k]] = param_combs[i][k]
            train_args = TrainingArguments(
                output_dir = model_dir,
                per_device_eval_batch_size = params['per_device_eval_batch_size'],
                num_train_epochs = params['num_train_epochs'],
                weight_decay = params['weight_decay'],
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                fp16=True,
                report_to=['tensorboard'],
                **param_args # KIV
            )
            model = Text_CLF(self.model_type).model        
            trainer = Trainer(
                model=model,
                args=train_args,
                train_dataset=self.core_ds['train'],
                eval_dataset=self.core_ds['test'],
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )
            trainer.args.dataloader_num_workers = 0
            trainer.args.dataloader_prefetch_factor = None
            trainer.train()
            metrics = trainer.evaluate(self.data.test_ds)
            results.append(metrics)
            if metrics['eval_loss'] < best_loss:
                best_loss = metrics['eval_loss']
                best_results = metrics
                best_params = param_args
            print(metrics)
        print('Best results:\n',best_params,'\n',best_results)
