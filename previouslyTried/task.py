import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import pandas as pd
import math

def euclidean_distance(vec1, vec2):
    vec3 = vec1 - vec2
    return math.sqrt(np.sum(vec3**2))

embeddings = {}

def load_glove(filename):
    print("Loading Glove Model")
    df = pd.read_csv(filename, sep=" ", quoting=3, header=None, index_col=0)
    global embeddings 
    embeddings = {key: val.values for key, val in df.T.items()}

def find_closest_embeddings(w):
    try:
        word = embeddings[w]
        return sorted(embeddings.keys(), key=lambda x: euclidean_distance(embeddings[x], word))
    except Exception as e:
        return ""

def get_closest_vector(w):
    try:
        e = find_closest_embeddings(w)[0]
        return embeddings[e]
    except Exception as e:
        return np.zeros(len(list(embeddings.values())[0]))

class BiLSTM(nn.Module):
  def __init__(self, input_size, input_dim, hidden_dim, word_embeddings):

    super(BiLSTM, self).__init__()

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.embeddings = nn.Embedding.from_pretrained(word_embeddings)
    self.classifier = nn.LSTM(input_size, input_dim, hidden_dim, bidirectional=True)

  def forward(self, sentence):
    u = self.embeddings(sentence)
    return self.classifier(u)

def train(training):
    model = BiLSTM(embeddings)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)
    epochs = 20
    for epoch in range(epochs):
        epoch_losses = list()
        for batch in training:
            optimizer.zero_grad()
            prediction = model(batch.text.T)
            loss = loss_function(prediction, batch.label)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print('train loss on epoch {} : {:.3f}'.format(epoch, np.mean(epoch_losses)))



load_glove("glove.6B.50d.txt")
#vals = np.array(list(embeddings.values()))
#print(find_closest_embeddings("she")[:10])


def load_data(files):
    train, dev, test = {},{},{}
    for i,file in enumerate(files):
        d = open(file, "r", encoding="utf-8")
        data_file = np.array([line for line in d.readlines() if line != "\n"][1:],dtype=object)
        for f in data_file:
            x = f.split()
            if i == 0:
                train = {x[0] : get_closest_vector(x[0])}
            elif i == 1:
                dev = {x[0] : get_closest_vector(x[0])}
            else:
                test = {x[0] : get_closest_vector(x[0])}
            
    return train, dev, test

train, dev, test = load_data(["data/train.conll","data/dev.conll","data/test.conll"])
print(np.array(train)[:5])