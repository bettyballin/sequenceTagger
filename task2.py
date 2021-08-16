import numpy as np
import pandas as pd
import torch
from torch import nn
from torchtext.legacy.data import Field, LabelField, BucketIterator
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import GloVe
from torch.optim import Adam


def load_data(file):
    d = open(file, "r", encoding="utf-8")
    data_file = np.array([line for line in d.readlines() if line != "\n"][1:],dtype=object)
    text_to_label = []
    for f in data_file:
        x = f.split()
        if not [x[0],x[-1]] in text_to_label:
            text_to_label.append([x[0],x[-1]])
    df_file = pd.DataFrame(text_to_label, columns=['text', 'label'])
    return df_file

train_df = load_data("data/train.conll")
dev_df = load_data("data/dev.conll")
test_df = load_data("data/test.conll")
df = train_df.append(dev_df.append(test_df))

text = Field(sequential=False, pad_token=0)
label = LabelField(sequential=False)

text.build_vocab(df['text'].apply(lambda x: text.preprocess(x)), vectors='glove.6B.50d')
label.build_vocab(df['label'].apply(lambda x: text.preprocess(x)))

glove = GloVe(name='6B', dim=50)

print(text.vocab.vectors.size())
print(glove.vectors.size())


class CustomTextDataset(Dataset):
    def __init__(self, txt, labels):
        self.labels = labels
        self.text = txt
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        txt = self.text[idx]
        sample = {"Text": txt, "Class": label}
        return sample

class BiLSTM(nn.Module):
  def __init__(self, input_size, input_dim, hidden_dim, word_embeddings):
    super(BiLSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.embeddings = nn.Embedding.from_pretrained(word_embeddings)
    self.classifier = nn.LSTM(input_size, input_dim, hidden_dim, bidirectional=True)

  def forward(self, x):
    print(x)
    print(self.embeddings)
    u = self.embeddings(x)
    print(u)
    return self.classifier(u)

model = BiLSTM(50, 100, 1, text.vocab.vectors)

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)
epochs = 20
f=(
    ('text', text),
    ('label', label)
)

train_ds = CustomTextDataset(train_df['text'], train_df['label'])
dev_ds = CustomTextDataset(dev_df['text'], dev_df['label'])
test_ds = CustomTextDataset(test_df['text'], test_df['label'])
print(Dataset.__subclasses__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
train_iter, dev_iter, test_iter = BucketIterator.splits(
    datasets=(train_ds, dev_ds, test_ds), 
    batch_sizes=(1, 1, 1),
    sort=False,
    device=device
)'''

train_loader = DataLoader(train_ds, batch_size=1)
test_loader = DataLoader(test_ds, batch_size=1)
p = nn.Embedding.from_pretrained(text.vocab.vectors)
d = p.parameters()

for epoch in range(epochs):
    epoch_losses = list()
    for t, label in train_df.items():
        optimizer.zero_grad()
        print(type(text.vocab.vectors[text.vocab[t]]))
        
        prediction = model(torch.tensor(text.vocab.vectors[text.vocab[t]]).to(torch.int64))
        loss = loss_function(prediction, label)

        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    print('train loss on epoch {} : {:.3f}'.format(epoch, np.mean(epoch_losses)))
    
    test_losses = list()
    for t, label in test_df.items():
        with torch.no_grad():
            optimizer.zero_grad()
            prediction = model(text.vocab.vectors[text.vocab[t]])
            loss = loss_function(prediction, label)
            
            test_losses.append(loss.item())
    
    print('test loss on epoch {}: {:.3f}'.format(epoch, np.mean(test_losses)))