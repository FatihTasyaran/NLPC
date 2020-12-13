#from transformers import *
import os

#tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
#model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

#def bertify_data(datapath):

folder_mask = '/home/fatih/Documents/CS546/Project/trial_data/machine-translation/0/'
info_units = folder_mask + 'info-units/'
stanza = folder_mask + '1406.1078v3-Stanza-out.txt'
entities = folder_mask + 'entities.txt'
sentences = folder_mask + 'sentences.txt'

##JSONS##
json_model = folder_mask + 'info-units/model.json'
jsons = os.listdir(info_units)
json_strings = []

print("JSONS:",jsons)

for item in jsons:
    reader = open(info_units+item)
    data = reader.read()
    json_strings.append(data)
##JSONS##

stanza_reader = open(stanza)
stanza_sentences = stanza_reader.readlines()

#json_reader = open(json_model)
#json_string = json_reader.read()

sentences_reader = open(sentences)
string_loc = sentences_reader.readlines()
int_loc = [int(x)-1 for x in string_loc] ##Because our list starts from zero :)

for item in int_loc:
    print(item+1, stanza_sentences[item])

for i in range(len(json_strings)):
    for loc in int_loc:
        found = json_strings[i].find(stanza_sentences[loc][2:-2]) ##Crop a bit
        if(found != -1):
            print(jsons[i] + '\n' + stanza_sentences[loc] + '\n##############')
            #print(loc+1)


                        
###
###FROM THAT POINT, WILL TURN STANZA LINES INTO A CSV, ALONG WITH CLASSES AS COLUMNS AND FOUND CONTRIBUTION CLASSES TAGGED WITH 1 THEN FEED INTO BERT



