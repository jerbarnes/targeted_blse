import xml.etree.ElementTree as ET
from collections import Counter
from nltk import word_tokenize

def get_semeval_sentence_data(file, binary=True):
    """
    The data is a tuple of (left context,
                            right context,
                            aspect term,
                            polarity)
    """
    if binary:
        label_map = {'positive':1, 'neutral': 1, 'negative': 0, 'conflict':1}
    else:
        label_map = {'positive':2, 'neutral': 1, 'negative': 0, 'conflict':1}
    	
    
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
                labels = [aspect.get('polarity') for aspect in aspects]
                majority_label = Counter(labels).most_common(n=1)[0][0]
                data.append((['<s>'] + word_tokenize(text) + ['</s>'], label_map[majority_label]))


    return data
