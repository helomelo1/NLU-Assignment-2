"""
PREPROCESSING FILE

Requirements:
WordCloud MatPlotLib
"""

import re
import json
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

stopwords = {
    "department","office","iit","jodhpur","contact",
    "home","index","portal","links","important",
    "copyright","feedback","policy",
    "yes","no","male","female","year","first","second",
    "one","two","three","also","may","must",
    "student","course","program","semester",
    "credit","requirement","academic"
}

def clean_text(text):
    text = text.lower()
    text = text.replace("ph.d.", "phd").replace("ph.d", "phd")
    text = text.replace("m.tech.", "mtech").replace("b.tech.", "btech")
    text = text.replace("m.sc.", "msc")
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    return tokens

def generate_bigrams(tokens, min_count=5):
    pairs = list(zip(tokens, tokens[1:]))
    counts = Counter(pairs)
    return [f"{w1}_{w2}" for (w1, w2), c in counts.items() if c >= min_count and w1 != w2]

files = ["ALL_html.txt", "ALL_pdf.txt"]

clean_docs = []
all_tokens = []

for file in files:
    text = open(file, encoding="utf-8").read()
    tokens = clean_text(text)
    bigrams = generate_bigrams(tokens, min_count=5)
    tokens = tokens + bigrams
    clean_docs.append(tokens)
    all_tokens.extend(tokens)

freq = Counter(all_tokens)
min_count = 5
filtered_tokens = [t for t in all_tokens if freq[t] >= min_count]

print("Documents:", len(clean_docs))
print("Total Tokens:", len(filtered_tokens))
print("Vocabulary Size:", len(set(filtered_tokens)))

with open("sentences.json", "w") as f:
    json.dump(clean_docs, f)

text = " ".join(filtered_tokens)

wc = WordCloud(width=1000, height=500, background_color="white", collocations=False).generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wc)
plt.axis("off")
plt.show()