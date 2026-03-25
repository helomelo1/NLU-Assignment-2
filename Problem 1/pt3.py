import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

# load tokenized sentences (each sentence is a list of words)
with open("sentences.json") as f:
    sentences = json.load(f)

# flatten all sentences into a single list of tokens
tokens = [w for s in sentences for w in s]

# build vocabulary and mappings between words and indices
vocab = list(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

# compute word frequencies and create a probability distribution for negative sampling
# raising to 0.75 reduces dominance of very frequent words
word_counts = Counter(tokens)
total = sum(word_counts.values())

prob_dist = np.array([word_counts[w] for w in vocab]) / total
prob_dist = prob_dist ** 0.75
prob_dist /= prob_dist.sum()

# sigmoid function used for binary classification (positive vs negative samples)
def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))

# sample k negative word indices based on probability distribution
def get_neg_samples(k):
    return np.random.choice(vocab_size, size=k, p=prob_dist)

# training function for both CBOW and Skip-gram
def train_word2vec(sg=0, embedding_dim=100, window_size=5, neg_samples=5, epochs=20, lr=0.025):
    # sg = 0 → CBOW, sg = 1 → Skip-gram

    # W: input embeddings, W_out: output embeddings
    W = np.random.randn(vocab_size, embedding_dim) * 0.01
    W_out = np.random.randn(vocab_size, embedding_dim) * 0.01

    for epoch in range(epochs):
        loss = 0

        # iterate through each sentence
        for sent in tqdm(sentences, desc=f"Epoch {epoch+1}/{epochs}"):
            indices = [word2idx[w] for w in sent if w in word2idx]

            # iterate over each target word in sentence
            for i, target in enumerate(indices):
                start = max(0, i - window_size)
                end = min(len(indices), i + window_size + 1)

                # context = words around target
                context = indices[start:i] + indices[i+1:end]

                if not context:
                    continue

                if sg == 0:
                    # CBOW: predict target from context words

                    # average context embeddings
                    h = np.mean(W[context], axis=0)

                    # positive sample update
                    score = sigmoid(np.dot(W_out[target], h))
                    loss += -np.log(score + 1e-9)

                    grad = (score - 1)

                    old_out = W_out[target].copy()
                    W_out[target] -= lr * grad * h

                    # distribute gradient equally among context words
                    for c in context:
                        W[c] -= lr * grad * old_out / len(context)

                    # negative sampling updates
                    negs = get_neg_samples(neg_samples)
                    for neg in negs:
                        score = sigmoid(np.dot(W_out[neg], h))
                        loss += -np.log(1 - score + 1e-9)

                        grad = score

                        old_out = W_out[neg].copy()
                        W_out[neg] -= lr * grad * h

                        for c in context:
                            W[c] -= lr * grad * old_out / len(context)

                else:
                    # Skip-gram: predict each context word from target

                    for ctx in context:
                        # positive sample update
                        score = sigmoid(np.dot(W_out[ctx], W[target]))
                        loss += -np.log(score + 1e-9)

                        grad = (score - 1)

                        old_out = W_out[ctx].copy()
                        old_in = W[target].copy()

                        W_out[ctx] -= lr * grad * old_in
                        W[target] -= lr * grad * old_out

                        # negative sampling updates
                        negs = get_neg_samples(neg_samples)
                        for neg in negs:
                            score = sigmoid(np.dot(W_out[neg], W[target]))
                            loss += -np.log(1 - score + 1e-9)

                            grad = score

                            old_out = W_out[neg].copy()
                            old_in = W[target].copy()

                            W_out[neg] -= lr * grad * old_in
                            W[target] -= lr * grad * old_out

        print(f"Model: {'SkipGram' if sg else 'CBOW'} | Epoch {epoch+1} | Loss {loss:.4f}")

    return W

# run experiments with different hyperparameters
configs = [
    (0, 300, 5, 10),
    (1, 300, 5, 10)
]

models = {}

for sg, dim, win, neg in configs:
    key = f"{'sg' if sg else 'cbow'}_d{dim}_w{win}_n{neg}"
    print("\nTraining:", key)

    embeddings = train_word2vec(
        sg=sg,
        embedding_dim=dim,
        window_size=win,
        neg_samples=neg
    )

    models[key] = embeddings

# cosine similarity between two vectors
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)

# find top-k nearest neighbors for a word
def nearest(embeddings, word, k=5):
    if word not in word2idx:
        return []

    idx = word2idx[word]
    sims = []

    for i in range(vocab_size):
        if i == idx:
            continue
        sims.append((idx2word[i], cosine(embeddings[idx], embeddings[i])))

    sims.sort(key=lambda x: -x[1])
    return sims[:k]

# analogy: a : b :: c : ?
def analogy(embeddings, a, b, c, k=5):
    if a not in word2idx or b not in word2idx or c not in word2idx:
        return []

    vec = embeddings[word2idx[b]] - embeddings[word2idx[a]] + embeddings[word2idx[c]]

    sims = []
    for i in range(vocab_size):
        sims.append((idx2word[i], cosine(vec, embeddings[i])))

    sims.sort(key=lambda x: -x[1])
    return sims[:k]

# choose one trained model for evaluation
embeddings = list(models.values())[0]

print("\nNearest Neighbors:\n")
for w in ["research", "student", "phd", "exam"]:
    print(w, ":", nearest(embeddings, w))

print("\nAnalogies:\n")
print("ug:btech :: pg: ?", analogy(embeddings, "ug", "btech", "pg"))
print("student:exam :: professor: ?", analogy(embeddings, "student", "exam", "professor"))
print("btech:mtech :: ug: ?", analogy(embeddings, "btech", "mtech", "ug"))

# select some words for visualization
words = ["research", "phd", "student", "exam", "professor", "assistant_professor"]

vecs = []
labels = []

for w in words:
    if w in word2idx:
        vecs.append(embeddings[word2idx[w]])
        labels.append(w)

vecs = np.array(vecs)

# reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(vecs)

plt.figure()
plt.scatter(reduced[:, 0], reduced[:, 1])

for i, label in enumerate(labels):
    plt.annotate(label, (reduced[i, 0], reduced[i, 1]))

plt.title("Word Embeddings (PCA Projection)")
plt.show()