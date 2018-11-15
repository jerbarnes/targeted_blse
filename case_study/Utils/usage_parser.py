import os
from nltk import sent_tokenize
from nltk import word_tokenize

def get_usage_opinions(usage_filename):

    text = open(usage_filename + '.txt').readlines()
    anns = open(usage_filename + '-a1.csv').readlines()
    rels = open(usage_filename + '-a1.rel').readlines()

    sents = {}

    for s in text:
        sent_id, _, _, _, title, sent = s.split('\t')
        sents[sent_id] = {}
        sents[sent_id]['text'] = title + ' ' + sent
        sents[sent_id]['aspects'] = {}

    for a in anns:
        dtype, sent_id, off1, off2, asp, asp_id, label, rel = a.split('\t')
        sents[sent_id]['aspects'][asp_id] = {}
        sents[sent_id]['aspects'][asp_id]['off1'] = off1
        sents[sent_id]['aspects'][asp_id]['off2'] = off2
        sents[sent_id]['aspects'][asp_id]['label'] = label
        sents[sent_id]['aspects'][asp_id]['aspect'] = asp

    data = []
    for rel in rels:
        _, sent_id, aspect_id, expression_id, aspect, expression = rel.split('\t')
        text = sents[sent_id]['text']
        aspect_off1 = int(sents[sent_id]['aspects'][aspect_id]['off1'])
        aspect_off2 = int(sents[sent_id]['aspects'][aspect_id]['off2'])
        label = sents[sent_id]['aspects'][expression_id]['label']
        l = text[:aspect_off1].replace('.', ' . ')
        try:
            l = sent_tokenize(l)[-1]
        except IndexError:
            pass
        r = text[aspect_off2:].replace('.', ' . ')
        try:
            r = sent_tokenize(r)[0]
        except IndexError:
            pass
        data.append((l, r, aspect, label))

    return data

def create_usage_training_example(l, r, target, polarity, binary):
    if binary:
        label_map = {'positive':1, 'negative':0}
    else:        
        label_map = {'positive':2, 'neutral':1, 'negative':0}
        
    l = ['<s>'] + l.lower().split()
    r = r.lower().split() + ['</s>']
    target = target.lower().split()
    y = label_map[polarity]
    
    return l, r, target, y


def get_usage_data(DIR, binary=True, lang='en'):
    targets = []

    if lang == 'en':
        filenames = ['coffeemachine', 'trashcan', 'vacuum',
                     'microwave', 'dishwasher', 'washer',
                     'toaster', 'cutlery']
    else:
        filenames = ['washer', 'microwave', 'vacuum', 'trashcan',
                     'toaster', 'coffeemachine', 'cutlery']
    
    for file in filenames:
        fname = lang + '-' + file
        try:
            targets.extend(get_usage_opinions(os.path.join(DIR, fname)))
        except IndexError:
            # print('{0}: no opinions'.format(file))
            pass

    training_examples = []
    for l, r, t, y in targets:
        try:
            training_examples.append(create_usage_training_example(l, r, t, y, binary=binary))
        except:
            pass
    return training_examples
