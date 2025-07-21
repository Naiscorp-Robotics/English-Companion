import pandas as pd 
import re 
import json 
def preprocess_data(df):
    data = df['dialog']
    dataset = []
    for text in data: 
        text = text.strip("[]")
        sentences = re.findall(r"'(.*?)'|\"(.*?)\"", text)
        clean_sentences = [s[0] if s[0] else s[1] for s in sentences]
        # remove empty sentences
        clean_sentences = [sentence.strip() for sentence in clean_sentences]
        # nối khoảng trắng xung quanh dấu ’
        clean_sentences = [re.sub(r"\s+’\s+", "’", text) for text in clean_sentences]
        # Loại bỏ khoảng trắng thừa trước dấu câu
        clean_sentences = [re.sub(r"\s+([.,?!])", r"\1", text) for text in clean_sentences]
        # Loại bỏ khoảng trắng thừa sau dấu nháy đơn nếu cần 
        clean_sentences = [re.sub(r"\s+'\s+", "'", text) for text in clean_sentences]
        clean_sentences = [re.sub(r'(\w)\.(\w)', r'\1. \2', text) for text in clean_sentences]
        # lower
        clean_sentences = [sentence.lower() for sentence in clean_sentences]
        dataset.append(clean_sentences)
    return dataset

DATA_PATH = "data/classification/raw/train.csv"

df = pd.read_csv(DATA_PATH)
dataset = preprocess_data(df)

with open("data/classification/correct/train.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
