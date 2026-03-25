import torch
import string
import os

from torch.nn.utils.rnn import pad_sequence
from models import VanillaRNN, BiLSTM, RNNAttention
import torch.nn.functional as F

# Loading names from TrainingNames file
training_file = "TrainingNames.txt"
with open(training_file, "r") as f:
    names = [line.strip().lower() for line in f if line.strip()]

# Ensuring Clean Data
names = [n for n in names if n.isalpha()]

# Construct Vocabulary
chars = sorted(list(set("".join(names)))) # Character level vocab
chars = ['<s>', '<e>'] + chars # Adding start and end char tags

# Preparing a dictionary for storing indices of each character
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
vocab_size = len(chars)

# Dataset
def encode(name):
    return [stoi['<s>']] + [stoi[c] for c in name] + [stoi['<e>']]

def decode(indices):
    return "".join([itos[i] for i in indices if itos[i] not in ['<s>', '<e>']])

def create_dataset(names):
    X, y = [], []

    for name in names:
        enc = encode(name)

        for i in range(len(enc) - 1):
            X.append(enc[:i+1])
            y.append(enc[i+1])

    return X, y

X, y = create_dataset(names)

# Training Loop
def train(model, X, y, epochs=10, lr=1e-3, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        
        perm = torch.randperm(len(X))
        
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            
            batch_X = [torch.tensor(X[j]) for j in idx]
            batch_Y = torch.tensor([y[j] for j in idx])
            
            batch_X = pad_sequence(batch_X, batch_first=True)
            
            # One-hot
            batch_X = F.one_hot(batch_X, num_classes=vocab_size).float()
            
            # MOVE TO DEVICE
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            out = model(batch_X)
            loss = criterion(out, batch_Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")


# Generating Names using the trained model
def generate(model, max_len=15, temperature=0.8):
    model.eval()
    with torch.no_grad():
        seq = [stoi['<s>']]
        
        for _ in range(max_len):
            x = torch.tensor([seq])
            x = torch.nn.functional.one_hot(x, num_classes=vocab_size).float().to(device)
            
            out = model(x)
            logits = out / temperature
            
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            
            # Prevent too short names
            if itos[idx] == '<e>' and len(seq) < 4:
                continue
            
            if itos[idx] == '<e>':
                break
            
            seq.append(idx)
        
        return decode(seq)

def novelty_rate(model, train_names, n_samples=200):
    generated = [generate(model) for _ in range(n_samples)]
    
    train_set = set(train_names)
    novel = [name for name in generated if name not in train_set]
    
    return len(novel) / n_samples


def diversity(model, n_samples=200):
    generated = [generate(model) for _ in range(n_samples)]
    
    unique = len(set(generated))
    return unique / n_samples


def evaluate(model, train_names):
    nov = novelty_rate(model, train_names)
    div = diversity(model)
    
    print(f"Novelty Rate: {nov:.3f}")
    print(f"Diversity: {div:.3f}")

# Hyperparameters
hidden_size = 256
learning_rate = 1e-3
epochs = 350
batch_size = 256

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Model dictionary with initialized instances
models = {
    "VanillaRNN": VanillaRNN(vocab_size, hidden_size, vocab_size),
    "BiLSTM": BiLSTM(vocab_size, hidden_size, vocab_size),
    "RNNAttention": RNNAttention(vocab_size, hidden_size, vocab_size)
}

for name, model in models.items():
    print(f"\n===== Using Model: {name} =====")
    
    model = model.to(device)
    
    # Move data to device inside training loop (modify train slightly if needed)
    train(model, X, y, epochs=epochs, lr=learning_rate, batch_size=batch_size)
    
    print("\n--- Generated Samples ---")
    for _ in range(10):
        print(generate(model))
    
    print("\n--- Evaluation ---")
    evaluate(model, names)

    torch.save(model.state_dict(), "model.pth")
    size_mb = os.path.getsize("model.pth") / (1024 * 1024)

    print(f"Model size: {size_mb:.2f} MB")

    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)