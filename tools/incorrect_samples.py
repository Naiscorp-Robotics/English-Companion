import re
import random
import json 
import sys
from numpy.testing import break_cycles
from openai import OpenAI
import os 
import string
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def remove_word_but_keep_punctuation(sentence):
    # Tách theo khoảng trắng, mỗi phần tử là 1 token có thể bao gồm cả dấu câu
    tokens = sentence.split()

    # Xác định token nào là từ (có ít nhất 1 ký tự chữ hoặc số)
    def is_word(token):
        return any(c.isalnum() for c in token)

    # Lấy chỉ số các token là từ
    word_indices = [i for i, token in enumerate(tokens) if is_word(token)]

    if len(word_indices) < 2:
        return None

    # Chọn ngẫu nhiên một từ để xóa
    index_to_remove = random.choice(word_indices)

    # Xóa từ đó
    del tokens[index_to_remove]

    # Trả lại câu sau khi xóa
    return ' '.join(tokens)

def move_random_word(sentence):
    tokens = sentence.split()

    # Hàm xác định token là từ (ít nhất 1 chữ hoặc số)
    def is_word(token):
        return any(c.isalnum() for c in token)

    word_indices = [i for i, token in enumerate(tokens) if is_word(token)]

    if len(word_indices) < 2:
        return None  # Không đủ từ để thực hiện

    # Chọn một từ ngẫu nhiên
    from_idx = random.choice(word_indices)

    # Chọn vị trí mới để chèn, khác vị trí hiện tại
    insert_positions = [i for i in range(len(tokens) + 1) if i != from_idx and i != from_idx + 1]
    to_idx = random.choice(insert_positions)

    # Lấy và xóa từ đó
    word = tokens.pop(from_idx)

    # Chèn lại vào vị trí mới
    tokens.insert(to_idx if to_idx <= from_idx else to_idx - 1, word)

    return ' '.join(tokens)


def change_word(sentence, OPENAI_API_KEY):
    # Tách câu theo khoảng trắng
    tokens = sentence.split()

    # Hàm kiểm tra token là từ (có ít nhất 1 ký tự chữ hoặc số)
    def is_word(token):
        return any(c.isalnum() for c in token)

    # Lấy chỉ số các token là từ
    word_indices = [i for i, token in enumerate(tokens) if is_word(token)]

    if len(word_indices) < 2:
        return None

    # Chọn ngẫu nhiên một từ để sửa
    idx_to_change = random.choice(word_indices)
    word_to_change = tokens[idx_to_change]

    mistake_types = ["grammatically", "semantically"]
    mistake_type = random.choice(mistake_types)
    
    # Tạo prompt chi tiết cho GPT
    prompt = (
        f"Original sentence tokens: {' '.join(tokens)}\n"
        f"Change only this word: '{word_to_change}' (at position {idx_to_change}).\n"
        f"Make sure the new sentence is incorrect ({mistake_type}).\n"
        f"Don't change any other tokens, including punctuation.\n"
        f"Return only the modified sentence, no explanations."
    )

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You modify a sentence by replacing one word to make it wrong. Keep everything else exactly the same."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

DATA_PATH = "data/classification/correct/train.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

random.seed(2025)
incorrect_samples = []
for i, dialog in enumerate(dataset): 
    if i < 300: 
        continue
    if i > 500: 
        break
    for sentence in dialog:
        in_samples = [] 
        in_samples.append(sentence)
        in_samples.append(remove_word_but_keep_punctuation(sentence))
        in_samples.append(move_random_word(sentence))
        in_samples.append(change_word(sentence, OPENAI_API_KEY))
        incorrect_samples.append(in_samples)
    if i % 20 == 0: 
        print(f"Processed {i} dialogs")
        with open("data/classification/incorrect/train_v2.json", "w", encoding="utf-8") as f:
            json.dump(incorrect_samples, f, ensure_ascii=False, indent=4)
