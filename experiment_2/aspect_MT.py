import sys, os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Utils.utils import*
from Utils.kaf_parser import *
from Utils.usage_parser import *
from Utils.Datasets import *
from Utils.WordVecs import *


class aspect_MT(nn.Module):
    
    def __init__(self, args, vecs,
                 output_dim=5):
        super(aspect_MT, self).__init__()

        self.emb = nn.Embedding(vecs.vocab_length, vecs.vector_size)
        self.emb.weight.data.copy_(torch.from_numpy(vecs._matrix))
        self.w2idx = vecs._w2idx
        self.idx2w = vecs._idx2w

        # Projection vectors
        self.m = nn.Linear(vecs.vector_size, vecs.vector_size, bias=False)
        # Classifier
        self.clf = nn.Linear(vecs.vector_size * 3, output_dim)
        # Loss Functions
        self.criterion = nn.CrossEntropyLoss()
        # Optimizer
        self.optim = torch.optim.Adam(self.parameters(), 
                                      lr= args.learning_rate, 
                                      weight_decay=args.weight_decay)

        # History
        self.history  = {'loss':[], 'dev_cosine':[], 'dev_f1':[], 'cross_f1':[],
                         'syn_cos':[], 'ant_cos':[], 'cross_syn':[], 'cross_ant':[]}
        self.emb.weight.requires_grad=False



    def dump_weights(self, outfile):
        w1 = self.m.weight.data.numpy()
        w3 = self.clf.weight.data.numpy()
        b = self.clf.bias.data.numpy()
        np.savez(outfile, w1, w2, w3, b)

    def load_weights(self, weight_file):
        f = np.load(weight_file)
        w1 = self.m.weight.data.copy_(torch.from_numpy(f['arr_0']))
        w3 = self.clf.weight.data.copy_(torch.from_numpy(f['arr_1']))
        b = self.clf.bias.data.copy_(torch.from_numpy(f['arr_2']))

    def idx_vecs(self, sentence, model):
        sent = []
        for w in sentence:
            try:
                sent.append(model[w])
            except:
                sent.append(0)
        return torch.LongTensor(np.array(sent))

    def lookup(self, X, model):
        return [self.idx_vecs(s, model) for s in X]

    def predict(self, left_x, right_x, term_x):
        left = self.ave_vecs(left_x)
        right = self.ave_vecs(right_x)
        term = self.ave_vecs(term_x)
        l_proj = self.m(left)
        r_proj = self.m(right)
        t_proj = self.m(term)
        x_proj = torch.cat((l_proj, t_proj, r_proj), dim=1)
        # out = F.softmax(self.clf(x_proj)) in pytorch 2.0 this gives an error
        out = F.softmax(self.clf(x_proj), dim=1)
        return out

    def ave_vecs(self, X, src=True):
        vecs = []
        if src:
            idxs = self.lookup(X, self.w2idx)
            for i in idxs:
                vecs.append(self.emb(Variable(i)).mean(0))
        else:
            idxs = self.lookup(X, self.w2idx)
            for i in idxs:
                vecs.append(self.emb(Variable(i)).mean(0))
        return torch.stack(vecs)

    def classification_loss(self, left_x, right_x, term_x, y):
        pred = self.predict(left_x, right_x, term_x)
        y = Variable(torch.LongTensor(y))
        loss = self.criterion(pred, y)
        return loss

    def fit(self, train_X, 
            dev_X,
            weight_dir='models',
            batch_size=100,
            epochs=100):
        num_batches = int(len(train_X) / batch_size)
        best_cross_f1 = 0
        num_epochs = 0
        for i in range(epochs):
            idx = 0
            num_epochs += 1
            for j in range(num_batches):
                l, r, t, y = zip(*train_X[idx:idx+batch_size])
                idx += batch_size
                self.optim.zero_grad()
                loss = self.classification_loss(l, r, t, y)
                loss.backward()
                self.optim.step()
            if i % 1 == 0:
                # check source dev f1
                l, r, t, y = zip(*dev_X)
                xp = self.predict(l, r, t).data.numpy().argmax(1)
                # macro f1
                dev_f1 = per_class_f1(y, xp).mean()

                
                sys.stdout.write('\r epoch {0} loss: {1:.3f}  src_f1: {2:.3f}'.format(
                    i, loss.data[0], dev_f1))
                sys.stdout.flush()
                self.history['loss'].append(loss.data[0])
                self.history['dev_f1'].append(dev_f1)
                

    def plot(self, title=None, outfile=None):
        h = self.history
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(h['loss'], label='loss')
        ax.plot(h['dev_f1'], label='source_f1', linestyle=':')
        ax.set_ylim(-.5, 1.4)
        ax.legend(
                loc='upper center', bbox_to_anchor=(.5, 1.05),
                ncol=3, fancybox=True, shadow=True)
        if title:
            ax.title(title)
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--src_lang', default='en')
    parser.add_argument('-tl', '--trg_lang', default='es')
    parser.add_argument('-src_da', '--src_dataset', default='opener')
    parser.add_argument('-trg_da', '--trg_dataset', default='opener')
    parser.add_argument('-se', '--embeddings', default='embeddings/blse/google.txt')
    parser.add_argument('-e', '--epochs', default=50, type=int)
    parser.add_argument('-bi', '--binary', default=True, type=str2bool)
    parser.add_argument('-bs', '--batch_size', default=100, type=int)
    parser.add_argument('-emb', '--embedding_dim', default=300, type=int)
    parser.add_argument('-hid', '--hidden_dim', default=100, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-wd', '--weight_decay', default=3e-5, type=float)
    parser.add_argument('-cuda', default=True, type=str2bool)
    parser.add_argument('-seed', default=123, type=int)
    args = parser.parse_args()

    print_args(args)
    args.cuda = args.cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print('Importing embeddings...')
    vecs = WordVecs(args.embeddings)

    synonyms1, synonyms2, neg = get_syn_ant(args.src_lang, vecs)
    cross_syn1, cross_syn2, cross_neg = get_syn_ant(args.trg_lang, vecs)
    pdataset = ProjectionDataset('lexicons/{0}_{1}.txt'.format(args.src_lang, args.trg_lang), vecs, vecs)

 
    print('Importing datasets...')

    # Get training, dev, and test data
    if args.src_dataset == 'opener':
        if args.binary:
            train_data, dev_data, test_data = open_dataset('datasets/OpeNER/preprocessed/binary/en')
        else:
            train_data, dev_data, test_data = open_dataset('datasets/OpeNER/preprocessed/multi/en')
    elif args.src_dataset == 'semeval':
        if args.binary:
            train_data, dev_data, test_data = open_dataset('datasets/semeval_2016_aspect-based/preprocessed/binary/en')
        else:
            train_data, dev_data, test_data = open_dataset('datasets/semeval_2016_aspect-based/preprocessed/multi/en')
    elif args.src_dataset == 'usage':
        if args.binary:
            train_data, dev_data, test_data = open_dataset('datasets/USAGE-corpus-with-text/preprocessed/binary/en')
        else:
            train_data, dev_data, test_data = open_dataset('datasets/USAGE-corpus-with-text/preprocessed/multi/en')

    if args.trg_dataset == 'opener':
        if args.trg_lang in ['ca', 'eu']:
            if args.binary:
                basedir = 'datasets/MultiBooked/translated/binary/'
            else:
                basedir = 'datasets/MultiBooked/translated/multi/'
        else:
            if args.binary:
                basedir = 'datasets/OpeNER/translated/binary/'
            else:
                basedir = 'datasets/OpeNER/translated/multi/'
        trg_test = open_test(os.path.join(basedir, args.trg_lang))
    elif args.trg_dataset == 'semeval':
        if args.binary:
            trg_test = open_test(os.path.join('datasets/semeval_2016_aspect-based/translated/binary', args.trg_lang))
        else:
            trg_test = open_test(os.path.join('datasets/semeval_2016_aspect-based/translated/multi', args.trg_lang))
    elif args.trg_dataset == 'usage':
        if args.binary:
            trg_test = open_test(os.path.join('datasets/USAGE-corpus-with-text/translated/binary', args.trg_lang))
        else:
            trg_test = open_test(os.path.join('datasets/USAGE-corpus-with-text/translated/multi', args.trg_lang))

    # Get the number of outputs for the classifier
    _, _, _, y = zip(*train_data)
    output_dim = len(set(y))

    print('Initializing model...')
    aMT = aspect_MT(args, vecs,
                         output_dim=output_dim)

    print('Training...')
    try:
        aMT.fit(train_data, dev_data,
                 epochs=args.epochs)
    except KeyboardInterrupt:
        print('stopping training early...')
    
    print()
    l, r, t, y = zip(*test_data)
    src_pred = aMT.predict(l, r, t).data.numpy().argmax(1)
    f1 = per_class_f1(y, src_pred)
    print(f1)
    print(f1.mean())
    print()

    trgl, trgr, trgt, trgy = zip(*trg_test)
    trg_pred = aMT.predict(trgl, trgr, trgt).data.numpy().argmax(1)
    cross_f1 = per_class_f1(trgy, trg_pred)
    print(cross_f1)
    print(cross_f1.mean())

    outfile = os.path.join('predictions', args.src_dataset, '{0}-{1}'.format(args.src_lang, args.trg_lang),
                            'aMT-binary:{0}_epochs:{1}_learningrate:{2}_weightdecay:{3}_batchsize:{4}.txt'.format(
                                args.binary, args.epochs, args.learning_rate, args.weight_decay, args.batch_size))
    print_prediction(trg_pred, outfile)

    plotfile=os.path.join('figures', args.src_dataset, '{0}-{1}'.format(args.src_lang, args.trg_lang),
                            'aMT-binary:{0}_epochs:{1}_learningrate:{2}_weightdecay:{3}_batchsize:{4}.pdf'.format(
                                args.binary, args.epochs, args.learning_rate, args.weight_decay, args.batch_size))
    os.makedirs(os.path.dirname(plotfile), exist_ok=True)
    aMT.plot(outfile=plotfile)

    cross_dev_f1 = 0
    print_results(args, cross_dev_f1, cross_f1.mean(), 
                  clf='aMT')
