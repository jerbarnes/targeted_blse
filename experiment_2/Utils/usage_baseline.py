import os
from nltk import sent_tokenize
from nltk import word_tokenize
from collections import Counter

def get_usage_opinions(usage_filename, level='sent', train=True):

    text = open(usage_filename + '.txt').readlines()
    anns = open(usage_filename + '-a1.csv').readlines()
    rels = open(usage_filename + '-a1.rel').readlines()

    sents = {}

    for s in text:
        sent_id, _, _, _, title, sent = s.split('\t')
        sents[sent_id] = {}
        sents[sent_id]['text'] = title + ' ' + sent
        sents[sent_id]['aspects'] = {}
        sents[sent_id]['max_label'] = Counter()

    for a in anns:
        dtype, sent_id, off1, off2, asp, asp_id, label, rel = a.split('\t')
        sents[sent_id]['aspects'][asp_id] = {}
        sents[sent_id]['aspects'][asp_id]['off1'] = off1
        sents[sent_id]['aspects'][asp_id]['off2'] = off2
        sents[sent_id]['aspects'][asp_id]['label'] = label
        sents[sent_id]['aspects'][asp_id]['aspect'] = asp

    # get the most common labels for the sent:
    for rel in rels:
        _, sent_id, aspect_id, expression_id, aspect, expression = rel.split('\t')
        label = sents[sent_id]['aspects'][expression_id]['label']
        sents[sent_id]['max_label'].update([label])

    data = []
    for rel in rels:
        _, sent_id, aspect_id, expression_id, aspect, expression = rel.split('\t')
        text = sents[sent_id]['text']
        sents[sent_id]['aspects'][aspect_id]['aspect'] = aspect
        sents[sent_id]['aspects'][aspect_id]['expression'] = expression
        aspect_off1 = int(sents[sent_id]['aspects'][aspect_id]['off1'])
        aspect_off2 = int(sents[sent_id]['aspects'][aspect_id]['off2'])
        if train:
            label = sents[sent_id]['max_label'].most_common(1)[0][0]
        else:
            label = sents[sent_id]['aspects'][expression_id]['label']
        sents[sent_id]['aspects'][aspect_id]['sent'] = text

        # if sentence-level, split at target, sent tokenize and keep closest parts
        if level == 'sent':
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
            text = l + aspect + r
        sents[sent_id]['aspects'][aspect_id]['sent'] = text
        data.append((text, aspect, label))

    return data

def create_usage_training_example(text, polarity, binary=False):
    if binary:
        label_map = {'positive':1, 'negative':0}
    else:        
        label_map = {'positive':2, 'neutral':1, 'negative':0}

    text = ['<s>'] + word_tokenize(text.lower()) + ['</s>']
    y = label_map[polarity]
    
    return text, y

def get_usage_data(DIR, binary=True, lang='en', level='sent', train=True):
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
            targets.extend(get_usage_opinions(os.path.join(DIR, fname), level, train))
        except IndexError:
            # print('{0}: no opinions'.format(file))
            pass

    training_examples = []
    for text, t, y in targets:
        try:
            training_examples.append(create_usage_training_example(text, y, binary=binary))
        except:
            pass
    return training_examples
