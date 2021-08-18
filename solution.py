from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn
from torchtext.legacy.data import Field, Dataset, BucketIterator, Example
from torch.optim import Adam
import matplotlib.pyplot as plt
import time


def load_data(filename, datafields):
    """Load words and labels from specified file location

    Args:
        filename (string)
        datafields (list)

    Returns:
        Dataset: Dataset with examples from words & labels
    """
    d = open(filename, "r", encoding="utf-8")
    file = [line for line in d.readlines()][2:]
    words = []
    labels = []
    examples = []
    for f in file:
        line = f.strip()
        if not line: # end of sentence
            examples.append(Example.fromlist([words, labels], datafields))
            words = []
            labels = []
        else:
            columns = line.split()
            words.append(columns[0])
            labels.append(columns[-1])
    return Dataset(examples, datafields)

   
class BiLSTMTagger(nn.Module):
    """ Pytorch Module for sequence tagging, using a bidirectional LSTM 
        and a pretrained embedding

    Args:
        text (Field)
        label (Field)
        emb_dim (int)

    Returns:
        Module
    """
    def __init__(self, text, label, emb_dim):
        super(BiLSTMTagger, self).__init__()
        self.n_labels = len(label.vocab)

        # apply pretrained vectors to Embedding
        self.embedding = nn.Embedding(len(text.vocab), emb_dim)
        self.embedding.weight = torch.nn.Parameter(text.vocab.vectors, requires_grad=False)

        # build bidirectional LSTM with linear output layer and cross entropy loss
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=100, num_layers=1, bidirectional=True)
        self.top_layer = nn.Linear(200, self.n_labels)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')

        # save the position of pad word & label
        self.pad_word_id = text.vocab.stoi[text.pad_token]
        self.pad_label_id = label.vocab.stoi[label.pad_token]
    
    def forward(self, sentence, labels):
        scores = self.compute_outputs(sentence)
        # flatten scores and labels to apply the vectors in loss function
        scores = scores.view(-1, self.n_labels)
        labels = labels.view(-1)
        return self.loss(scores, labels)
            
    def compute_outputs(self, sentence):
        embedded = self.embedding(sentence) # look up the embedding for all words in the sentence
        lstm_out, _ = self.lstm(embedded)   # apply the LSTM
        out = self.top_layer(lstm_out)      # apply the linear top layer

        # add a large number in the column of pad tokens
        pad_mask = (sentence == self.pad_word_id).float()
        out[:, :, self.pad_label_id] += pad_mask*10000
        return out

    def predict(self, sentence):
        scores = self.compute_outputs(sentence)
        predicted = scores.argmax(dim=2)    # get the best prediction
        return np.array(predicted.t().cpu())

def to_spans(labels, voc):
    """ Convert a list of predicted labels to spans

    Args:
        labels (list): IDs of labels
        voc (list): Vocab of labels

    Returns:
        Dict: dictionary of spans
    """
    spans = {}
    current_lbl = None
    current_start = None
    for i, id in enumerate(labels):
        l = voc[id]
        if l[0] == 'B': 
            if current_lbl:
                spans[current_start] = (current_lbl, i)
            current_lbl = l[2:]
            current_start = i
        elif l[0] == 'I':
            if current_lbl:
                if current_lbl != l[2:]:
                    spans[current_start] = (current_lbl, i)
                    current_lbl = l[2:]
                    current_start = i
            else:
                current_lbl = l[2:]
                current_start = i
        else:
            if current_lbl:
                spans[current_start] = (current_lbl, i)
                current_lbl = None
                current_start = None
    return spans

def compare(gold, pred, stats):
    """Compares gold and prediction spans and store the result in spans

    Args:
        gold (dict): IDs of labels
        pred (dict): Vocab of labels
        stats (dict):

    Returns:
        Dict: dictionary of spans
    """
    for start, (lbl, end) in gold.items():
        stats['total']['gold'] += 1
        stats[lbl]['gold'] += 1
    for start, (lbl, end) in pred.items():
        stats['total']['pred'] += 1
        stats[lbl]['pred'] += 1
    for start, (glbl, gend) in gold.items():
        if start in pred:
            plbl, pend = pred[start]
            if glbl == plbl and gend == pend:
                stats['total']['corr'] += 1
                stats[glbl]['corr'] += 1

def evaluate_iob(predicted, gold, label_field, stats):
    """ Evaluate 

    Args:
        labels (list): IDs of labels
        voc (list): Vocab of labels

    Returns:
        Dict: dictionary of spans
    """
    gold_cpu = np.array(gold.t().cpu())
    gold_cpu = list(gold_cpu.reshape(-1))
    pred_cpu = [l for sen in predicted for l in sen]
    gold_spans = to_spans(gold_cpu, label_field.vocab.itos)
    pred_spans = to_spans(pred_cpu, label_field.vocab.itos)
    compare(gold_spans, pred_spans, stats)

def compute_f1(stats):
    """ Computes the f1 score

    Args:
        stats (dict): dictionary of predictions

    Returns:
        double: f1 score
    """
    if stats['pred'] == 0:
        return 0.0
    precision = stats['corr']/stats['pred']
    recall = stats['corr']/stats['gold']
    if precision > 0 and recall > 0:
        return 2*precision*recall/(precision+recall)
    else:
        return 0.0

# Create Fields for text and labels
TEXT = Field(init_token='<bos>', eos_token='<eos>', sequential=True, lower=False)
LABEL = Field(init_token='<bos>', eos_token='<eos>', sequential=True, unk_token=None)
fields = [('text', TEXT), ('label', LABEL)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
train_examples = load_data('data/train.conll', fields)
test_examples = load_data('data/test.conll', fields)

# Compute the number of text tokens for each dataset
n_tokens_train = 0
n_sentences_train = 0
for ex in train_examples:
    n_tokens_train += len(ex.text) + 2
    n_sentences_train += 1
n_tokens_test = 0       
for ex in test_examples:
    n_tokens_test += len(ex.text)

# Use GLOVE embeddings to build the vocabulary
TEXT.build_vocab(train_examples, vectors="glove.6B.50d")
LABEL.build_vocab(train_examples)
n_labels = len(LABEL.vocab)

# Create the model
model = BiLSTMTagger(TEXT, LABEL, emb_dim=50)
model.to(device)

# Set batch size
batch_size = 1
n_batches = n_sentences_train
mean_n_tokens = n_tokens_train / n_batches

# Create Iterators for each dataset
train_iterator = BucketIterator(
    train_examples,
    device=device,
    batch_size=batch_size,
    sort_key=lambda x: len(x.text),
    repeat=False,
    train=True,
    sort=True)
test_iterator = BucketIterator(
    test_examples,
    device=device,
    batch_size=batch_size,
    sort_key=lambda x: len(x.text),
    repeat=False,
    train=False,
    sort=True)

# Create the Adam optimizer
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# Set epochs to 20
n_epochs = 2

result = defaultdict(list)  

print('Start training the model')
# Train the model and print the loss and F1 score for each epoch
for i in range(1, n_epochs + 1):
    loss_sum = 0
    model.train()
    for batch in list(train_iterator):
        loss = model(batch.text, batch.label) / mean_n_tokens
        optimizer.zero_grad()            
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    train_loss = loss_sum / n_batches
    result['train_loss'].append(train_loss)

    stats = defaultdict(Counter)
    model.eval()
    with torch.no_grad():
        for batch in list(test_iterator):
            predicted = model.predict(batch.text)        

            # evaluation of the predicted labels           
            evaluate_iob(predicted, batch.label, LABEL, stats)

    f1 = compute_f1(stats['total'])
    result['f1'].append(f1)
    print(f'Epoch {i}: F1: {f1:.4f}, train loss = {train_loss:.4f}')

print('Evaluation on the test data:')
f1 = compute_f1(stats['total'])
print(f'F1 = {f1:.4f}')
for label in stats:
    if label != 'total':
        f1 = compute_f1(stats[label])
        print(f'{label:4s}: F1 = {f1:.4f}')

plt.plot(result['train_loss'])
plt.plot(result['f1'])
plt.legend(['training loss', 'F1-score'])
plt.show()