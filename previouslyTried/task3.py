from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchtext
from torchtext.legacy.data import Field, LabelField, Dataset, BucketIterator, Iterator, Example
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from torch.optim import Adam
import matplotlib.pyplot as plt
import time


def load_data(file, datafields):
    d = open(file, "r", encoding="utf-8")
    data_file = [line for line in d.readlines()][2:]
    words = []
    examples = []
    labels = []
    for f in data_file:
        line = f.strip()
        if not line:
            examples.append(Example.fromlist([words, labels], datafields))
            words = []
            labels = []
        else:
            columns = line.split()
            words.append(columns[0])
            labels.append(columns[-1])
    return Dataset(examples, datafields)
    
class BiLSTM(nn.Module):
    def __init__(self, text_field, label_field, emb_dim, size, update_pretrained=False):
        super(BiLSTM, self).__init__()
        voc_size = len(text_field.vocab)
        self.n_labels = len(label_field.vocab)
        self.embedding = nn.Embedding(voc_size, emb_dim)
        if text_field.vocab.vectors is not None:
            self.embedding.weight = torch.nn.Parameter(text_field.vocab.vectors, 
                                                        requires_grad=update_pretrained)

        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=size, num_layers=1, bidirectional=True)
        self.top_layer = nn.Linear(2*size, self.n_labels)
        self.pad_word_id = text_field.vocab.stoi[text_field.pad_token]
        self.pad_label_id = label_field.vocab.stoi[label_field.pad_token]
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
    
    def forward(self, sentences, labels):
        # shape: (max_len, n_sentences, n_labels).
        scores = self.compute_outputs(sentences)
        
        # Flatten the outputs and the gold-standard labels, to compute the loss.
        # The input to this loss needs to be one 2-dimensional and one 1-dimensional tensor.
        scores = scores.view(-1, self.n_labels)
        labels = labels.view(-1)
        return self.loss(scores, labels)

            
    def compute_outputs(self, sentences):
        # The words in the documents are encoded as integers. The shape of the documents
        # tensor is (max_len, n_docs), where n_docs is the number of documents in this batch,
        # and max_len is the maximal length of a document in the batch.

        # First look up the embeddings for all the words in the documents.
        # The shape is now (max_len, n_sentences, emb_dim).   
        embedded = self.embedding(sentences)

#
        # The shape of the output tensor is (max_len, n_sentences, 2*rnn_size).
        lstm_out, _ = self.lstm(embedded)
        
        # Apply the linear output layer.
        # The shape of the output tensor is (max_len, n_sentences, n_labels).
        out = self.top_layer(lstm_out)
        
        # Find the positions where the token is a dummy padding token.
        pad_mask = (sentences == self.pad_word_id).float()

        # For these positions, we add some large number in the column corresponding
        # to the dummy padding label.
        out[:, :, self.pad_label_id] += pad_mask*10000

        return out

    def predict(self, sentences):
        # Compute the outputs from the linear units.
        scores = self.compute_outputs(sentences)

        # Select the top-scoring labels. The shape is now (max_len, n_sentences).
        predicted = scores.argmax(dim=2)

        # We transpose the prediction to (n_sentences, max_len), and convert it
        # to a NumPy matrix.
        return predicted.t().cpu().numpy()

# Convert a list of BIO labels, coded as integers, into spans identified by a beginning, an end, and a label.
# To allow easy comparison later, we store them in a dictionary indexed by the start position.
def to_spans(l_ids, voc):
    spans = {}
    current_lbl = None
    current_start = None
    for i, l_id in enumerate(l_ids):
        l = voc[l_id]

        if l[0] == 'B': 
            # Beginning of a named entity: B-something.
            if current_lbl:
                # If we're working on an entity, close it.
                spans[current_start] = (current_lbl, i)
            # Create a new entity that starts here.
            current_lbl = l[2:]
            current_start = i
        elif l[0] == 'I':
            # Continuation of an entity: I-something.
            if current_lbl:
                # If we have an open entity, but its label does not
                # correspond to the predicted I-tag, then we close
                # the open entity and create a new one.
                if current_lbl != l[2:]:
                    spans[current_start] = (current_lbl, i)
                    current_lbl = l[2:]
                    current_start = i
            else:
                # If we don't have an open entity but predict an I tag,
                # we create a new entity starting here even though we're
                # not following the format strictly.
                current_lbl = l[2:]
                current_start = i
        else:
            # Outside: O.
            if current_lbl:
                # If we have an open entity, we close it.
                spans[current_start] = (current_lbl, i)
                current_lbl = None
                current_start = None
    return spans

# Compares two sets of spans and records the results for future aggregation.
def compare(gold, pred, stats):
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

# This function combines the auxiliary functions we defined above.
def evaluate_iob(predicted, gold, label_field, stats):
    # The gold-standard labels are assumed to be an integer tensor of shape
    # (max_len, n_sentences), as returned by torchtext.
    gold_cpu = gold.t().cpu().numpy()
    gold_cpu = list(gold_cpu.reshape(-1))

    # The predicted labels assume the format produced by pytorch-crf, so we
    # assume that they have been converted into a list already.
    # We just flatten the list.
    pred_cpu = [l for sen in predicted for l in sen]
    
    # Compute spans for the gold standard and prediction.
    gold_spans = to_spans(gold_cpu, label_field.vocab.itos)
    pred_spans = to_spans(pred_cpu, label_field.vocab.itos)

    # Finally, update the counts for correct, predicted and gold-standard spans.
    compare(gold_spans, pred_spans, stats)

# Computes precision, recall and F-score, given a dictionary that contains
# the counts of correct, predicted and gold-standard items.
def prf(stats):
    if stats['pred'] == 0:
        return 0, 0, 0
    p = stats['corr']/stats['pred']
    r = stats['corr']/stats['gold']
    if p > 0 and r > 0:
        f = 2*p*r/(p+r)
    else:
        f = 0
    return p, r, f


class Tagger:
    
    def __init__(self, lower):
        self.TEXT = Field(init_token='<bos>', eos_token='<eos>', sequential=True, lower=lower)
        self.LABEL = Field(init_token='<bos>', eos_token='<eos>', sequential=True, unk_token=None)
        self.fields = [('text', self.TEXT), ('label', self.LABEL)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def tag(self, sentences):
        # This method applies the trained model to a list of sentences.
        # First, create a torchtext Dataset containing the sentences to tag.
        examples = []
        for sen in sentences:
            labels = ['?']*len(sen) # placeholder
            examples.append(Example.fromlist([sen, labels], self.fields))
        dataset = Dataset(examples, self.fields)
        
        iterator = Iterator(
            dataset,
            device=self.device,
            batch_size=1,
            repeat=False,
            train=False,
            sort=False)
        
        # Apply the trained model to all batches.
        out = []
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                # Call the model's predict method. This returns a list of NumPy matrix
                # containing the integer-encoded tags for each sentence.
                predicted = self.model.predict(batch.text)

                # Convert the integer-encoded tags to tag strings.
                for tokens, pred_sen in zip(sentences, predicted):
                    out.append([self.LABEL.vocab.itos[pred_id] for _, pred_id in zip(tokens, pred_sen[1:])])
        return out
                
    def train(self):
        # Read training and validation data according to the predefined split.
        train_examples = load_data('data/train.conll', self.fields)
        valid_examples = load_data('data/test.conll', self.fields)
       # Count the number of words and sentences.
        n_tokens_train = 0
        n_sentences_train = 0
        for ex in train_examples:
            n_tokens_train += len(ex.text) + 2
            n_sentences_train += 1
        n_tokens_valid = 0       
        for ex in valid_examples:
            n_tokens_valid += len(ex.text)

        # Load the pre-trained embeddings that come with the torchtext library.
        use_pretrained = True
        if use_pretrained:
            print('We are using pre-trained word embeddings.')
            self.TEXT.build_vocab(train_examples, vectors="glove.6B.50d")
        else:  
            print('We are training word embeddings from scratch.')
            self.TEXT.build_vocab(train_examples, max_size=5000)
        self.LABEL.build_vocab(train_examples)
    
        # Create one of the models defined above.
        self.model = BiLSTM(self.TEXT, self.LABEL, emb_dim=50, size=100, update_pretrained=False)
    
        self.model.to(self.device)
    
        batch_size = 1
        n_batches = np.ceil(n_sentences_train / batch_size)

        mean_n_tokens = n_tokens_train / n_batches

        train_iterator = BucketIterator(
            train_examples,
            device=self.device,
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            train=True,
            sort=True)

        valid_iterator = BucketIterator(
            valid_examples,
            device=self.device,
            batch_size=64,
            sort_key=lambda x: len(x.text),
            repeat=False,
            train=False,
            sort=True)
    
        train_batches = list(train_iterator)
        valid_batches = list(valid_iterator)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-5)

        n_labels = len(self.LABEL.vocab)

        history = defaultdict(list)    
        
        n_epochs = 20
        
        for i in range(1, n_epochs + 1):

            t0 = time.time()

            loss_sum = 0

            self.model.train()
            for batch in train_batches:
                
                # Compute the output and loss.
                loss = self.model(batch.text, batch.label) / mean_n_tokens
                
                optimizer.zero_grad()            
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            train_loss = loss_sum / n_batches
            history['train_loss'].append(train_loss)

            # Evaluate on the validation set.
            if i % 1 == 0:
                stats = defaultdict(Counter)

                self.model.eval()
                with torch.no_grad():
                    for batch in valid_batches:
                        # Predict the model's output on a batch.
                        predicted = self.model.predict(batch.text)                   
                        # Update the evaluation statistics.
                        evaluate_iob(predicted, batch.label, self.LABEL, stats)
            
                # Compute the overall F-score for the validation set.
                _, _, val_f1 = prf(stats['total'])
                
                history['val_f1'].append(val_f1)
            
                t1 = time.time()
                print(f'Epoch {i}: train loss = {train_loss:.4f}, val f1: {val_f1:.4f}, time = {t1-t0:.4f}')
           
        # After the final evaluation, we print more detailed evaluation statistics, including
        # precision, recall, and F-scores for the different types of named entities.
        print()
        print('Final evaluation on the validation set:')
        p, r, f1 = prf(stats['total'])
        print(f'Overall: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')
        for label in stats:
            if label != 'total':
                p, r, f1 = prf(stats[label])
                print(f'{label:4s}: P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}')
        
        plt.plot(history['train_loss'])
        plt.plot(history['val_f1'])
        plt.legend(['training loss', 'validation F-score'])

tagger = Tagger(lower=False)
tagger.train()
plt.show()