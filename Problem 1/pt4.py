import json

# load preprocessed sentences (list of lists of tokens)
with open("sentences.json") as f:
    sentences = json.load(f)

# write each sentence as a single line in corpus.txt
with open("corpus.txt", "w") as f:
    for sent in sentences:
        # join tokens back into a sentence string
        line = " ".join(sent)
        
        # write sentence to file
        f.write(line + "\n")