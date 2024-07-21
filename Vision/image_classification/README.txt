# Overview:
'''
Arguments:
--phase: (default) train, test
--train_data_dir: (default) ./data/train/
--test_data_dir: (default) ./data/test/   
--model_dir: (default) model.pkl

Advised folder structure:
- PATH
  - data
    - train
      - **Image class folders
    - test
      - **Image class folders
  - main.py
  - model.pkl
  - requirements.txt
  - README.txt

* IMPORTANT: 
- Before running the script:
  - Ensure that all the packages in requirements.txt have been installed.
  - Change your directory in the command line to the folder concerned.
- Image class names and numbers follow the hw3 specifications exactly (including all irregular casing).
- The data folder structure is based on the default data_dir values provided. If your data is stored elsewhere, please set the data directories accordingly in command line, using the arguments specified above.
- The script assumes that both the training and test set directories have image class folders as direct child folders, as per the folder structure of the original dataset given to us.
- The sample code below for training and testing assumes that the dataset has already been split into 'train' and test' folders. If it has not, you may run the following line of code to split the dataset:
python -c "from main import *; generate_train_test_sets(r'data')"
'''

# To train model:
python -m main

# To test model:
python -m main --phase test