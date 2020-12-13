#from transformers import *
import os
import sys
import pandas as pd
import numpy as np

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
    #datapath = sys.argv[1]
    datapath = "/home/fatih/Documents/CS546/Project/trial_data/"
    sentences_dict = {}
    
    print(data)
    
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

    print(data)
    data.to_csv('tryer.csv')



                        
###
###FROM THAT POINT, WILL TURN STANZA LINES INTO A CSV, ALONG WITH CLASSES AS COLUMNS AND FOUND CONTRIBUTION CLASSES TAGGED WITH 1 THEN FEED INTO BERT



