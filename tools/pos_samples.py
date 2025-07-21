import spacy 
import json 

DATA_PATH = "data/classification/correct/train.json" 
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

pos_dataset = []
nlp = spacy.load("en_core_web_sm") 
for dialog in dataset: 
    pos_dialog = []
    for sentence in dialog: 
        pos_sentence = []
        doc = nlp(sentence)
        for token in doc: 
            pos_sentence.append((token.text, token.tag_))
        pos_dialog.append(pos_sentence)
    pos_dataset.append(pos_dialog)

with open("data/classification/correct/train_pos.json", "w", encoding="utf-8") as f:
    json.dump(pos_dataset, f, ensure_ascii=False, indent=4)
