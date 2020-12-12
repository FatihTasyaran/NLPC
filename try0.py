#from transformers import *
import os

#tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
#model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

folder_mask = '/home/fatih/Documents/CS546/Project/trial_data/machine-translation/0/'
stanza = folder_mask + '1406.1078v3-Stanza-out.txt'
entities = folder_mask + 'entities.txt'
sentences = folder_mask + 'sentences.txt'

##JSONS##
json_model = folder_mask + 'info-units/model.json'
jsons = os.listdir(folder_mask+'info-units/')
#print(jsons)


##JSONS##

stanza_reader = open(stanza)
stanza_sentences = stanza_reader.readlines()

json_reader = open(json_model)
json_string = json_reader.read()

sentences_reader = open(sentences)
string_loc = sentences_reader.readlines()
int_loc = [int(x) for x in string_loc]

print(int_loc)

for loc in int_loc:
    found = json_string.find(stanza_sentences[loc])
    if(found != -1):
        print(loc)
    

#print("json string:\n", json_string)



#print("at error:", stanza_sentences[26][220])



#counter = 1
#for item in stanza_sentences:
#    if(counter < 50):
#        print(counter, "->", item)
#    counter += 1


