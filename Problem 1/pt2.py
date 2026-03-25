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

def clean_sentence(text):
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

sentences = []
all_tokens = []

for file in files:
    text = open(file, encoding="utf-8").read()

    # split into sentences FIRST
    raw_sentences = re.split(r'[.!?]', text)

    for s in raw_sentences:
        tokens = clean_sentence(s)

        if len(tokens) < 4:
            continue

        bigrams = generate_bigrams(tokens, min_count=5)
        tokens = tokens + bigrams

        sentences.append(tokens)
        all_tokens.extend(tokens)

freq = Counter(all_tokens)
min_count = 5

# filter vocab AFTER building sentences
filtered_sentences = []
for sent in sentences:
    filtered = [t for t in sent if freq[t] >= min_count]
    if len(filtered) >= 4:
        filtered_sentences.append(filtered)

print("Sentences:", len(filtered_sentences))
print("Total Tokens:", sum(len(s) for s in filtered_sentences))
print("Vocabulary Size:", len(set(t for s in filtered_sentences for t in s)))

with open("sentences.json", "w") as f:
    json.dump(filtered_sentences, f)

text = " ".join(t for s in filtered_sentences for t in s)

wc = WordCloud(width=1000, height=500, background_color="white", collocations=False).generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wc)
plt.axis("off")
plt.show()