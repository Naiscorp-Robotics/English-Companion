import pandas as pd 
import re 

def preprocess_data(df):
    data = df['dialog']
    dataset = []
    for text in data: 
        text = text.strip("[]")
        sentences = re.findall(r"'(.*?)'|\"(.*?)\"", text)
        clean_sentences = [s[0] if s[0] else s[1] for s in sentences]
        dataset.append(clean_sentences)
    return dataset

df = pd.read_csv('data/classification/correct/train.csv') 
dataset = preprocess_data(df)
# for i,text in enumerate(dataset): 
#     if i == 10: 
#         break
#     for sentence in text: 
#         print(sentence)
#     print("-"*100)

import spacy

nlp = spacy.load("en_core_web_sm")

pos_dict = {}

for i,dialog in enumerate(dataset):
    for sentence in dialog:
        doc = nlp(sentence)
        for token in doc:
            pos_dict[token.pos_] = pos_dict.get(token.pos_, 0) + 1

print(pos_dict)
        
