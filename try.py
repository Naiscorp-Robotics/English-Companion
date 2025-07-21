import spacy

# Load mô hình tiếng Anh
nlp = spacy.load("en_core_web_sm")

sentence = "The woman who works here is from Japan, and she will meet me"

# Xử lý câu
doc = nlp(sentence)

# In ra từng từ và POS tag
for token in doc:
    print(f"{token.text}\t{token.pos_}\t{token.tag_}")

