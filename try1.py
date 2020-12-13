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
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=len(label_cols))
    num_labels = len(label_cols)

    optimizer = AdamW(model.parameters(),lr=2e-5)
                        
    max_length = 100
    encodings = tokenizer.batch_encode_plus(sentences, max_length=max_length, pad_to_max_length=True, truncation=True)
    print('Tokenizer outputs:', encodings.keys())

    input_ids = encodings['input_ids']
    token_type_ids = encodings['token_type_ids']
    attention_masks = encodings['attention_mask']

    train_inputs, validation_inputs, train_labels, validation_labels, train_token_types, validation_token_types, train_masks, validation_masks = train_test_split(input_ids, labels, token_type_ids,attention_masks,random_state=2020, test_size=0.10)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    train_token_types = torch.tensor(train_token_types)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)
    validation_token_types = torch.tensor(validation_token_types)

    batch_size = 32

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_token_types)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_token_types)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    torch.save(validation_dataloader,'validation_data_loader')
    torch.save(train_dataloader,'train_data_loader')
    
    
    ##TRAIN
    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 3

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0 #running loss
        nb_tr_examples, nb_tr_steps = 0, 0
  
        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_token_types = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()

            # # Forward pass for multiclass classification
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]

            # Forward pass for multilabel classification
            #outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            #logits = outputs[0]
            #loss_func = BCEWithLogitsLoss() 
            #loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
            # loss_func = BCELoss() 
            # loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
            #train_loss_set.append(loss.item())    

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        ##VALIDATION
        model.eval()

        # Variables to gather full output
        logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

        # Predict
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_token_types = batch
            with torch.no_grad():
                # Forward pass
                outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)

                b_logit_pred = b_logit_pred.detach().cpu().numpy()
                pred_label = pred_label.to('cpu').numpy()
                b_labels = b_labels.to('cpu').numpy()

            tokenized_texts.append(b_input_ids)
            logit_preds.append(b_logit_pred)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

        # Flatten outputs
        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        # Calculate Accuracy
        threshold = 0.50
        pred_bools = [pl>threshold for pl in pred_labels]
        true_bools = [tl==1 for tl in true_labels]
        val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')*100
        val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100

        print('F1 Validation Accuracy: ', val_f1_accuracy)
        print('Flat Validation Accuracy: ', val_flat_accuracy)

