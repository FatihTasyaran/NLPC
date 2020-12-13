from transformers import *
import os
import sys
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split

##FROM PETAL##
import tensorflow as tf
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
#from transformers import *
from tqdm import tqdm, trange
from ast import literal_eval
##FROM PETAL##


#tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
#model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

column_names = ['id', 'sentence', 'research-problem', 'approach', 'model', 'code', 'dataset', 'experimental-setup', 'hyperparameters', 'baselines', 'results', 'tasks', 'experiments', 'ablation-analysis']

empty_column = {'id':0,
                'sentence':'NE',
                'research-problem':0,
                'approach':0,
                'model':0,
                'code':0,
                'dataset':0,
                'experimental-setup':0,
                'hyperparameters':0,
                'baselines':0,
                'results':0,
                'tasks':0,
                'experiments':0,
                'ablation-analysis':0}

column_locs = {'research-problem': 2,
               'approach': 3,
               'model': 4,
               'code': 5,
               'dataset': 6,
               'experimental-setup': 7,
               'hyperparameters': 8,
               'baselines': 9,
               'results': 10,
               'tasks': 11,
               'experiments': 12,
               'ablation-analysis':13
}

ids = 0

data = pd.DataFrame(columns = column_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        #self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l1 = model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 6)
        
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for epoch in range(EPOCHS):
        train(epoch)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    

def bertify_data(path, sentences_dict):
    global ids
    global data
    
    print("path: ", path)
    info_units = path + 'info-units/'
    
    ##FIND STANZA OUT##
    files = os.listdir(path)
    stanza = ''
    for item in files:
        if item.find('Stanza') != -1:
            stanza = path+item
    #print('Stanza:', stanza)
    ##FIND STANZA OUT##
    
    entities = path + 'entities.txt'
    sentences = path + 'sentences.txt'
    
    ##JSONS##
    jsons = os.listdir(info_units)
    column_names = [it.split('.json')[0] for it in jsons]
    json_strings = []

    #print("JSONS:",jsons)
        
    for item in jsons:
        reader = open(info_units+item)
        jdata = reader.read()
        json_strings.append(jdata)
    ##JSONS##

    stanza_reader = open(stanza)
    stanza_sentences = stanza_reader.readlines()

    for item in stanza_sentences:
        sentences_dict[item] = ids
        sentence_frame = pd.DataFrame([empty_column])
        sentence_frame['id'] = ids
        sentence_frame['sentence'] = item
        data = data.append(sentence_frame, ignore_index=True)
        #data = pd.concat([data,sentence_frame], axis=1)
        ids += 1

    sentences_reader = open(sentences)
    string_loc = sentences_reader.readlines()
    int_loc = [int(x)-1 for x in string_loc] ##Because our list starts from zero :)

    #for item in int_loc:
    #   print(item+1, stanza_sentences[item])

    for item in jsons:
        json_reader = open(info_units+item)
        json_string = json_reader.read()

        
    for i in range(len(json_strings)):
        for loc in int_loc:
            found = json_strings[i].find(stanza_sentences[loc][2:-2]) ##Crop a bit
            print('#######')
            print(json_strings[i])
            print(column_names[i])
            print('#######')
            if(found != -1):
                #print(jsons[i] + '\n' + stanza_sentences[loc] + '\n##############')
                #print('#######')
                data.iat[sentences_dict[stanza_sentences[loc]], column_locs[column_names[i]]] = 1 
                ##print('Changing: ', sentences_dict[stanza_sentences[loc]], i)
            #print(loc+1)
            
            
def sub_context(path, sentences_dict):
    directories = os.listdir(path)
    #print(directories)

    for item in directories:
        bertify_data(path+item+'/', sentences_dict)


if __name__ == "__main__":

    try:
        data = pd.read_csv('tryer.csv')
        print("Data is read from csv")
    except:
        #datapath = sys.argv[1]
        datapath = "/home/fatih/Documents/CS546/Project/trial_data/"
        sentences_dict = {}
        
        ##GET CONTEXTS AND PATHS##
        contexts = os.listdir(datapath)
    
        for item in contexts:
            if(not os.path.isdir(datapath+item)):
                contexts.remove(item)

        paths = [datapath+it+'/' for it in contexts]
        #for item in paths:
        #    print(item)
        ##GET CONTEXTS AND PATHS##    

        for path in paths:
            sub_context(path, sentences_dict)

        
        data.to_csv('tryer.csv')



    print('average sentence length: ', data.sentence.str.split().str.len().mean())
    print('stdev sentence length: ', data.sentence.str.split().str.len().std())
    
    
    label_cols = list(column_names[2:])
    #print(label_cols)

    print('Count of 1 per label: \n', data[label_cols].sum(), '\n') # Label counts, may need to downsample or upsample
    print('Count of 0 per label: \n', data[label_cols].eq(0).sum())

    data['one_hot_labels'] = list(data[label_cols].values)
    print(data['one_hot_labels'])

    labels = list(data.one_hot_labels.values)
    sentences = list(data.sentence.values)
    
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    #model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=len(label_cols))
    num_labels = len(label_cols)

    MAX_LEN=100
    TRAIN_BATCH_SIZE=8
    VALID_BATCH_SIZE=4
    EPOCHS=1
    LEARNING_RATE=1e-05

    # Creating the dataset and dataloader for the neural network
    train_size = 0.8
    train_dataset=new_df.sample(frac=train_size,random_state=200)
    test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)


    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = BERTClass()
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    
