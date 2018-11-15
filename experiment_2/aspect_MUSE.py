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


class aspect_MUSE(nn.Module):
    
    def __init__(self, args, src_vecs, trg_vecs,
                 pdataset,
                 src_syn1, src_syn2, src_neg,
                 trg_syn1, trg_syn2, trg_neg,
                 output_dim=5):
        super(aspect_MUSE, self).__init__()
        
        # Embedding matrices
        self.semb = nn.Embedding(src_vecs.vocab_length, src_vecs.vector_size)
        self.semb.weight.data.copy_(torch.from_numpy(src_vecs._matrix))
        self.sw2idx = src_vecs._w2idx
        self.sidx2w = src_vecs._idx2w
        self.temb = nn.Embedding(trg_vecs.vocab_length, trg_vecs.vector_size)
        self.temb.weight.data.copy_(torch.from_numpy(trg_vecs._matrix))
        self.tw2idx = trg_vecs._w2idx
        self.tidx2w = trg_vecs._idx2w
        # Projection vectors
        self.m = nn.Linear(src_vecs.vector_size, src_vecs.vector_size, bias=False)
        # Classifier
        self.clf = nn.Linear(src_vecs.vector_size * 3, output_dim)
        # Loss Functions
        self.criterion = nn.CrossEntropyLoss()
        # Optimizer
        self.optim = torch.optim.Adam(self.parameters(), 
                                      lr= args.learning_rate, 
                                      weight_decay=args.weight_decay)
        # Datasets
        self.pdataset = pdataset
        self.src_syn1 = src_syn1
        self.src_syn2 = src_syn2
        self.src_neg = src_neg
        self.trg_syn1 = trg_syn1
        self.trg_syn2 = trg_syn2
        self.trg_neg = trg_neg
        # History
        self.history  = {'loss':[], 'dev_cosine':[], 'dev_f1':[], 'cross_f1':[],
                         'syn_cos':[], 'ant_cos':[], 'cross_syn':[], 'cross_ant':[]}
        self.semb.weight.requires_grad=False
        self.temb.weight.requires_grad=False


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
        
    def project(self, x, y):
        """
        Project into shared space.
        """
        x_lookup = torch.LongTensor(np.array([self.sw2idx[w] for w in x]))
        y_lookup = torch.LongTensor(np.array([self.tw2idx[w] for w in y]))
        x_embedd = self.semb(Variable(x_lookup))
        y_embedd = self.temb(Variable(y_lookup))
        x_proj = self.m(x_embedd)
        y_proj = self.m(y_embedd)
        return x_proj, y_proj

    def project_one(self, x, src=True):
        if src:
            x_lookup = torch.LongTensor(np.array([self.sw2idx[w] for w in x]))
            x_embedd = self.semb(Variable(x_lookup))
            x_proj = self.m(x_embedd)
        else:
            x_lookup = torch.LongTensor(np.array([self.tw2idx[w] for w in x]))
            x_embedd = self.temb(Variable(x_lookup))
            x_proj = self.m(x_embedd)
        return x_proj

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

    def predict(self, left_x, right_x, term_x, src=True):
        left = self.ave_vecs(left_x, src)
        right = self.ave_vecs(right_x, src)
        term = self.ave_vecs(term_x, src)

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
            idxs = self.lookup(X, self.sw2idx)
            for i in idxs:
                vecs.append(self.semb(Variable(i)).mean(0))
        else:
            idxs = self.lookup(X, self.tw2idx)
            for i in idxs:
                vecs.append(self.temb(Variable(i)).mean(0))
        return torch.stack(vecs)

    def classification_loss(self, left_x, right_x, term_x, y, src=True):
        pred = self.predict(left_x, right_x, term_x, src=src)
        y = Variable(torch.LongTensor(y))
        loss = self.criterion(pred, y)
        return loss

    def fit(self, train_X, 
            dev_X,
            trg_dev_X,
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
                # check cosine distance between dev translation pairs
                xdev = self.pdataset._Xdev
                ydev = self.pdataset._ydev
                xp, yp = self.project(xdev, ydev)
                score = cos(xp, yp)

                
                # check source dev f1
                l, r, t, y = zip(*dev_X)
                xp = self.predict(l, r, t).data.numpy().argmax(1)
                # macro f1
                dev_f1 = per_class_f1(y, xp).mean()


                # check target dev f1
                l, r, t, y = zip(*trg_dev_X)
                xp = self.predict(l, r, t, src=False).data.numpy().argmax(1)
                # macro f1
                cross_f1 = per_class_f1(y, xp).mean()

                """
                print('loss: {0:.3f} trans: {1:.3f} dev f1: {2:.3f} cross f1: {3:.3f}'.format(loss.data[0], score.data[0], dev_f1, cross_f1))
                
                
                if cross_f1 > best_cross_f1:
                    best_cross_f1 = cross_f1
                    weight_file = os.path.join(weight_dir, '{0}epochs-{1}batchsize-{2}alpha-{3:.3f}crossf1'.format(num_epochs, batch_size, alpha, best_cross_f1))
                    self.dump_weights(weight_file)
                """
                

                # check cosine distance between source sentiment synonyms
                p1 = self.project_one(self.src_syn1)
                p2 = self.project_one(self.src_syn2)
                syn_cos = cos(p1, p2)

                # check cosine distance between source sentiment antonyms
                p3 = self.project_one(self.src_syn1)
                n1 = self.project_one(self.src_neg)
                ant_cos = cos(p3, n1)

                # check cosine distance between target sentiment synonyms
                cp1 = self.project_one(self.trg_syn1, src=False)
                cp2 = self.project_one(self.trg_syn2, src=False)
                cross_syn_cos = cos(cp1, cp2)

                # check cosine distance between target sentiment antonyms
                cp3 = self.project_one(self.trg_syn1, src=False)
                cn1 = self.project_one(self.trg_neg, src=False)
                cross_ant_cos = cos(cp3, cn1)
                
                sys.stdout.write('\r epoch {0} loss: {1:.3f}  trans: {2:.3f}  src_f1: {3:.3f}  trg_f1: {4:.3f}  src_syn: {5:.3f}  src_ant: {6:.3f}  cross_syn: {7:.3f}  cross_ant: {8:.3f}'.format(
                    i, loss.data[0], score.data[0], dev_f1, cross_f1, syn_cos.data[0],
                    ant_cos.data[0], cross_syn_cos.data[0], cross_ant_cos.data[0]))
                sys.stdout.flush()
                self.history['loss'].append(loss.data[0])
                self.history['dev_cosine'].append(score.data[0])
                self.history['dev_f1'].append(dev_f1)
                self.history['cross_f1'].append(cross_f1)
                self.history['syn_cos'].append(syn_cos.data[0])
                self.history['ant_cos'].append(ant_cos.data[0])
                self.history['cross_syn'].append(cross_syn_cos.data[0])
                self.history['cross_ant'].append(cross_ant_cos.data[0])
                

    def plot(self, title=None, outfile=None):
        h = self.history
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(h['dev_cosine'], label='translation_cosine')
        ax.plot(h['dev_f1'], label='source_f1', linestyle=':')
        ax.plot(h['cross_f1'], label='target_f1', linestyle=':')
        ax.plot(h['syn_cos'], label='source_synonyms', linestyle='--')
        ax.plot(h['ant_cos'], label='source_antonyms', linestyle='-.')
        ax.plot(h['cross_syn'], label='target_synonyms', linestyle='--')
        ax.plot(h['cross_ant'], label='target_antonyms', linestyle='-.')
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
    parser.add_argument('-se', '--src_embeddings', default='embeddings/muse/EN-ES/vectors-en.txt')
    parser.add_argument('-te', '--trg_embeddings', default='embeddings/muse/EN-ES/vectors-es.txt')
    parser.add_argument('-e', '--epochs', default=50, type=int)
    parser.add_argument('-bi', '--binary', default=True, type=str2bool)
    parser.add_argument('-bs', '--batch_size', default=100, type=int)
    parser.add_argument('-emb', '--embedding_dim', default=300, type=int)
    parser.add_argument('-hid', '--hidden_dim', default=100, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('-wd', '--weight_decay', default=0.0, type=float)
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
    src_vecs = WordVecs(args.src_embeddings)
    trg_vecs = WordVecs(args.trg_embeddings)

    synonyms1, synonyms2, neg = get_syn_ant(args.src_lang, src_vecs)
    cross_syn1, cross_syn2, cross_neg = get_syn_ant(args.trg_lang, trg_vecs)
    pdataset = ProjectionDataset('lexicons/{0}-{1}.txt'.format(args.src_lang, args.trg_lang), src_vecs, trg_vecs)


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
                basedir = 'datasets/MultiBooked/preprocessed/binary/'
            else:
                basedir = 'datasets/MultiBooked/preprocessed/multi/'
        else:
            if args.binary:
                basedir = 'datasets/OpeNER/preprocessed/binary/'
            else:
                basedir = 'datasets/OpeNER/preprocessed/multi/'
        trg_train, trg_dev, trg_test = open_dataset(os.path.join(basedir, args.trg_lang))
    elif args.trg_dataset == 'semeval':
        if args.binary:
            trg_train, trg_dev, trg_test = open_dataset(os.path.join('datasets/semeval_2016_aspect-based/preprocessed/binary', args.trg_lang))
        else:
            trg_train, trg_dev, trg_test = open_dataset(os.path.join('datasets/semeval_2016_aspect-based/preprocessed/multi', args.trg_lang))
    elif args.trg_dataset == 'usage':
        if args.binary:
            trg_train, trg_dev, trg_test = open_dataset(os.path.join('datasets/USAGE-corpus-with-text/preprocessed/binary', args.trg_lang))
        else:
            trg_train, trg_dev, trg_test = open_dataset(os.path.join('datasets/USAGE-corpus-with-text/preprocessed/multi', args.trg_lang))

    # Get the number of outputs for the classifier
    _, _, _, y = zip(*train_data)
    output_dim = len(set(y))

    print('Initializing model...')
    aMUSE = aspect_MUSE(args, src_vecs, trg_vecs, pdataset,
                         synonyms1, synonyms2, neg,
                         cross_syn1, cross_syn2, cross_neg,
                         output_dim=output_dim)

    print('Training...')
    try:
        aMUSE.fit(train_data, dev_data,
                 trg_dev,
                 epochs=args.epochs)
    except KeyboardInterrupt:
        print('stopping training early...')
    
    print()
    l, r, t, y = zip(*test_data)
    src_pred = aMUSE.predict(l, r, t, src=True).data.numpy().argmax(1)
    f1 = per_class_f1(y, src_pred)
    print(f1)
    print(f1.mean())
    print()

    trgl, trgr, trgt, trgy = zip(*trg_test)
    trg_pred = aMUSE.predict(trgl, trgr, trgt, src=False).data.numpy().argmax(1)
    cross_f1 = per_class_f1(trgy, trg_pred)
    print(cross_f1)
    print(cross_f1.mean())

    outfile = os.path.join('predictions', args.src_dataset, '{0}-{1}'.format(args.src_lang, args.trg_lang),
                            'aMUSE-binary:{0}_epochs:{1}_learningrate:{2}_weightdecay:{3}_batchsize:{4}.txt'.format(
                                args.binary, args.epochs, args.learning_rate, args.weight_decay, args.batch_size))
    print_prediction(trg_pred, outfile)

    plotfile=os.path.join('figures', args.src_dataset, '{0}-{1}'.format(args.src_lang, args.trg_lang),
                            'aMUSE-binary:{0}_epochs:{1}_learningrate:{2}_weightdecay:{3}_batchsize:{4}.pdf'.format(
                                args.binary, args.epochs, args.learning_rate, args.weight_decay, args.batch_size))
    os.makedirs(os.path.dirname(plotfile), exist_ok=True)
    aMUSE.plot(outfile=plotfile)

    cross_dev_f1 = aMUSE.history['cross_f1'][-1]
    print_results(args, cross_dev_f1, cross_f1.mean(), 
                  clf='aMUSE')
