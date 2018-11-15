import xml.etree.ElementTree as ET
import sys, os
import numpy as np
from sklearn.metrics import log_loss, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from scipy.spatial.distance import cosine
import torch.nn as nn
import re

def print_args(args):
    for arg in vars(args):
        print('{0}:\t{1}'.format(arg, getattr(args, arg)))
    print()


def print_prediction(prediction, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        for i in prediction:
            f.write("{0}\n".format(i))

def print_results(args, dev_f1, test_f1, 
                  clf='aBLSE'):
    outfile = os.path.join('results', args.src_dataset, '{0}-{1}'.format(args.src_lang, args.trg_lang),
                            '{0}-binary:{1}.txt'.format(
                                clf, args.binary))
    if clf in ['sentBLSE', 'aBLSE', 'aBLSE_target', 'aBLSE_weighted']:
        header = "Epochs\tLR\tWD\tBS\talpha\tDev F1\tTest F1\n"
        body = "{0}\t{1}\t{2}\t{3}\t{4}\t{5:0.3f}\t{6:0.3f}\n".format(
                args.epochs, args.learning_rate, args.weight_decay, 
                args.batch_size, args.alpha, dev_f1, test_f1)
    else:
        header = "Epochs\tLR\tWD\tBS\tDev F1\tTest F1\n"
        body = "{0}\t{1}\t{2}\t{3}\t{4:0.3f}\t{5:0.3f}\n".format(
                args.epochs, args.learning_rate, args.weight_decay, 
                args.batch_size, dev_f1, test_f1)


    if not os.path.exists(outfile):
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        with open(outfile, 'w') as f:
            f.write(header)

    with open(outfile, "a") as f:
        f.write(body)


def open_dataset(datadir, lang='en', mt=False):
    if lang == 'en':
        train, dev, test = [], [], []
        for line in open(os.path.join(datadir, 'train.txt')):
            l, r, t, y = line.strip().split('|')
            train.append((l.split(), r.split(), t.split(), int(y)))
        for line in open(os.path.join(datadir, 'dev.txt')):
            l, r, t, y = line.strip().split('|')
            dev.append((l.split(), r.split(), t.split(), int(y)))
        for line in open(os.path.join(datadir, 'test.txt')):
            l, r, t, y = line.strip().split('|')
            test.append((l.split(), r.split(), t.split(), int(y)))
        return train, dev, test
    else:
        test = []
        if mt:
            for line in open(os.path.join(datadir, 'translated.txt')):
                l, r, t, y = line.strip().split('|')
                test.append((l.split(), r.split(), t.split(), int(y)))
        else:
            for line in open(os.path.join(datadir, 'test.txt')):
                l, r, t, y = line.strip().split('|')
                test.append((l.split(), r.split(), t.split(), int(y)))
        return test

def get_semeval_data(file, labels=['positive', 'negative']):
    """
    The data is a tuple of (left context,
                            right context,
                            aspect term,
                            polarity)
    """
    if 'neutral' in labels:
        label_map = {'positive':2, 'neutral': 1, 'negative': 0, 'conflict':1}
    else:
        label_map = {'positive':1, 'neutral': 1, 'negative': 0, 'conflict':1}
    
    tree = ET.parse(file)
    root = tree.getroot()
    
    data = []
    
    for review in root:
        children = review.getchildren()
        sents = children[0].getchildren()
        for sent in sents:
            ch = sent.getchildren()
            if len(ch) == 2:
                text = ch[0].text
                aspects = ch[1].getchildren()
                for aspect in aspects:
                    term = aspect.get('target').lower().split()
                    label = aspect.get('polarity')
                    polarity = label_map[label]
                    if label in labels:
                        from_ = int(aspect.get('from'))
                        to_ = int(aspect.get('to'))
                        # TODO: add preprocessing here
                        # The <s> and </s> tags are problematic. Because of the class imbalance, they
                        # are highly correlated with positive sentiment but
                        # it's necessary to have them, as sometimes they are the
                        # only context we have to the left or right.
                        left_context = ['<s>'] + text[:from_].lower().split()
                        right_context = text[to_:].lower().split() + ['</s>']
                        #left_context = text[:from_].lower().split()
                        #right_context = text[to_:].lower().split()
                        data.append((left_context, right_context, term, polarity))

    return data

def to_array(X, n=2):
    """
    Converts a list scalars to an array of size len(X) x n
    >>> to_array([0,1], n=2)
    >>> array([[ 1.,  0.],
               [ 0.,  1.]])
    """
    return np.array([np.eye(n)[x] for x in X])

def per_class_f1(y, pred):
    """
    Returns the per class f1 score.
    Todo: make this cleaner.
    """
    
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    
    results = []
    for j in range(num_classes):
        class_y = y[:,j]
        class_pred = pred[:,j]
        f1 = f1_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results)

def per_class_prec(y, pred):
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    results = []
    for j in range(num_classes):
        class_y = y[:,j]
        class_pred = pred[:,j]
        f1 = precision_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results)

def per_class_rec(y, pred):
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    results = []
    for j in range(num_classes):
        class_y = y[:,j]
        class_pred = pred[:,j]
        f1 = recall_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results)


class ProjectionDataset():
    """
    A wrapper for the translation dictionary. The translation dictionary
    should be word to word translations separated by a tab. The
    projection dataset only includes the translations that are found
    in both the source and target vectors.
    """
    def __init__(self, translation_dictionary, src_vecs, trg_vecs):
        (self._Xtrain, self._Xdev, self._ytrain,
         self._ydev) = self.getdata(translation_dictionary, src_vecs, trg_vecs)

    def getdata(self, translation_dictionary, src_vecs, trg_vecs):
        x, y = [], []
        with open(translation_dictionary) as f:
            for line in f:
                src, trg = line.split()
                try:
                    _ = src_vecs[src]
                    _ = trg_vecs[trg]
                    x.append(src)
                    y.append(trg)
                except:
                    pass
        xtr, xdev = train_dev_split(x)
        ytr, ydev = train_dev_split(y)
        return xtr, xdev, ytr, ydev

def train_dev_split(x, train=.9):
    # split data into training and development, keeping /train/ amount for training.
    train_idx = int(len(x)*train)
    return x[:train_idx], x[train_idx:]

def cos(x, y):
    """
    This returns the mean cosine similarity between two sets of vectors.
    """
    c = nn.CosineSimilarity()
    return c(x,y).mean()

def get_syn_ant(lang, vecs):
    # This is a quick way to import the sentiment synonyms and antonyms to check their behaviour during training.
    synonyms1 = [l.strip() for l in open(os.path.join('syn-ant', lang, 'syn1.txt')) if l.strip() in vecs._w2idx]
    synonyms2 = [l.strip() for l in open(os.path.join('syn-ant', lang, 'syn2.txt')) if l.strip() in vecs._w2idx]
    neg = [l.strip() for l in open(os.path.join('syn-ant', lang, 'neg.txt')) if l.strip() in vecs._w2idx]
    idx = min(len(synonyms1), len(synonyms2), len(neg))
    return synonyms1[:idx], synonyms2[:idx], neg[:idx]

def get_best_run(weightdir):
    """
    This returns the best dev f1, parameters, and weights from the models
    found in the weightdir.
    """
    best_params = []
    best_f1 = 0.0
    best_weights = ''
    for file in os.listdir(weightdir):
        epochs = int(re.findall('[0-9]+', file.split('-')[-4])[0])
        batch_size = int(re.findall('[0-9]+', file.split('-')[-3])[0])
        alpha = float(re.findall('0.[0-9]+', file.split('-')[-2])[0])
        f1 = float(re.findall('0.[0-9]+', file.split('-')[-1])[0])
        if f1 > best_f1:
            best_params = [epochs, batch_size, alpha]
            best_f1 = f1
            weights = os.path.join(weightdir, file)
            best_weights = weights
    return best_f1, best_params, best_weights

def str2bool(v):
    # Converts a string to a boolean, for parsing command line arguments
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
