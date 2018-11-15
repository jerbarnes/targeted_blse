from lxml import etree
from lxml.etree import fromstring
import os

parser = etree.XMLParser(recover=True, encoding='utf8')

def get_opinions(kaf):
        xml = open(kaf).read().encode('utf8')
        root = fromstring(xml, parser)

        # unfortunately, all of the datasets are slightly different
        # so we have to set up some differences here
        if 'OpeNER/en' in kaf:
                idx = 'wid'
                text_idx = 1
                opinion_idx = 7
        elif 'OpeNER/es' in kaf:
                idx = 'wid'
                text_idx = 1
                opinion_idx = 6
        elif 'MultiBooked/eu' in kaf:
                idx = 'id'
                text_idx = 1
                opinion_idx = 4
        elif 'MultiBooked/ca' in kaf:
                idx = 'wid'
                text_idx = 3
                opinion_idx = 5
                
        
        word_forms = root[text_idx].getchildren()
        opinions = root[opinion_idx].getchildren()

        tokens = {}
                
        for w in word_forms:
            tokens[w.get(idx)] = {}
            tokens[w.get(idx)]['wfm'] = w.text
            tokens[w.get(idx)]['sent'] = w.get('sent')

        num_sents = []
        for w in word_forms:
                sidx = w.get('sent')
                if sidx not in num_sents:
                        num_sents.append(sidx)

        sents = dict([(str(k), {}) for k in num_sents])
        
        for w in word_forms:
                sents[w.get('sent')][w.get(idx)] = w.text

        target_list = []

        for opinion in opinions:
            try:
                targets = opinion[2][1].getchildren()
                targets = [t.get('id') for t in targets]
                targets = [l.replace('t', 'w') for l in targets]
                sent_id = tokens[targets[0]]['sent']
                polarity = opinion[3].get('polarity')
                skeys = list(sents[sent_id].keys())
                if 'MultiBooked/ca' in kaf:
                        s = ['w_'+str(j) for j in sorted([int(i[2:]) for i in skeys])]
                else:
                        s = ['w'+str(j) for j in sorted([int(i[1:]) for i in skeys])]
                sent = [sents[sent_id][i] for i in s]
                target = [tokens[i]['wfm'] for i in targets]
                #print(' '.join(target))
                #print(polarity)
                #print(' '.join(sent))
                #print()
                target_list.append((target, sent, polarity))

            except:
                pass

        return target_list


def create_training_example(target, sent, polarity, binary):
        if binary:
                label_map = {'StrongPositive':1, 'Positive':1, 'Negative':0, 'StrongNegative':0}
        else:        
                label_map = {'StrongPositive':3, 'Positive':2, 'Negative':1, 'StrongNegative':0}
        
        start_idx = sent.index(target[0])
        end_idx = sent.index(target[-1])
        
        l = ['<s>'] + sent[:start_idx]
        r = sent[end_idx+1:] + ['</s>']
        y = label_map[polarity]
        return l, r, target, y

def get_opener_data(DIR, binary=True):
        targets = []
        for file in os.listdir(DIR):
                # print(file)
                try:
                        targets.extend(get_opinions(os.path.join(DIR, file)))
                except IndexError:
                        # print('{0}: no opinions'.format(file))
                        pass

        training_examples = []
        for t in targets:
                try:
                        training_examples.append(create_training_example(*t, binary=binary))
                except:
                        pass
        return training_examples
