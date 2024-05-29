from src.model_building import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data.csv', help='directory for data file')
    parser.add_argument('--model_type', default='bert', help='model type: bert | distilbert | roberta')
    parser.add_argument('--prev_turns', default=True, help='(bool) to incorporate the previous turns of conversations or not')
    parser.add_argument('--cross_val', default=False, help='(bool) to train with cross validation or not')
    # parser.add_argument('--model_dir', default='bert', help='directory for exported model file') # Default: if model_type = roberta, model_dir = 'roberta'
    parser.add_argument('--mode', default='tune', help='train|evaluate|tune. if evaluate, the script simply loads a trained model for evaluation')
    opt = parser.parse_args()
    model = Model(opt.data_dir,opt.model_type,opt.prev_turns)
    model_dir = opt.model_type
    if opt.mode!='tune':
        if opt.mode == 'train':
            print('Training:')
            model.train(model_dir,opt.cross_val)
        print('Evaluating:')
        model.evaluate(model_dir)
    else:
        model.tune_param(model_dir)