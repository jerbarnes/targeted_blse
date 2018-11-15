from lxml import etree
from lxml.etree import fromstring
import os
from collections import Counter

parser = etree.XMLParser(recover=True, encoding='utf8')

def get_tag_idx(root, tag):
    for i, e in enumerate(root):
        if e.tag == tag:
            return i

def get_opinions(kaf):
        xml = open(kaf).read().encode('utf8')
        root = fromstring(xml, parser)

        # unfortunately, all of the datasets are slightly different
        # so we have to set up some differences here
        if 'OpeNER/en' in kaf:
                idx = 'wid'
        elif 'OpeNER/es' in kaf:
                idx = 'wid'
        elif 'MultiBooked/eu' in kaf:
                idx = 'id'
        elif 'MultiBooked/ca' in kaf:
                idx = 'wid'

        text_idx = get_tag_idx(root, 'text')
        opinion_idx = get_tag_idx(root, 'opinions')
         
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

        sents = dict([(str(k), {'tokens':{}, 'opinions':{}}) for k in num_sents])
        
        for w in word_forms:
                sents[w.get('sent')]['tokens'][w.get(idx)] = w.text

        for opinion in opinions:
            try:
                targets = opinion[2][1].getchildren()
                targets = [t.get('id') for t in targets]
                targets = [l.replace('t', 'w') for l in targets]
                sent_id = tokens[targets[0]]['sent']
                polarity = opinion[3].get('polarity')
                skeys = list(sents[sent_id]['tokens'].keys())
                if 'MultiBooked/ca' in kaf:
                        s = ['w_'+str(j) for j in sorted([int(i[2:]) for i in skeys])]
                else:
                        s = ['w'+str(j) for j in sorted([int(i[1:]) for i in skeys])]
                sent = [sents[sent_id]['tokens'][i] for i in s]
                target = [tokens[i]['wfm'] for i in targets]
                t = ' '.join(target)
                sents[sent_id]['opinions'][t] = {}
                sents[sent_id]['opinions'][t]['label'] = polarity
                sents[sent_id]['opinions'][t]['text'] = sent
            except:
                #print('passing')
                pass

        #return sents
        
        training_data = []
        for sent_id in sents.keys():
            labels = [l['label'] for l in sents[sent_id]['opinions'].values()]
            c = Counter(labels)
            try:
                majority_label = c.most_common(n=1)[0][0]
                target = list(sents[sent_id]['opinions'].keys())[0]
                text = sents[sent_id]['opinions'][target]['text']
                training_data.append((text, majority_label))
            except IndexError:
                pass

        return training_data
    
def get_opener_sentence_data(DIR, binary=True):
    data = []
    if binary:
        label_map = {'StrongPositive':1, 'Positive':1, 'Negative':0, 'StrongNegative':0}
    else:        
        label_map = {'StrongPositive':3, 'Positive':2, 'Negative':1, 'StrongNegative':0}
        
    for f in os.listdir(DIR):
        try:
            data.extend(get_opinions(os.path.join(DIR, f)))
        except TypeError:
            print(f)

    data = [(['<s>'] + t + ['</s>'], label_map[l]) for t, l in data]
    return data
