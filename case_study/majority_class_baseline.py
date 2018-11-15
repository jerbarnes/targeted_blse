from Utils.utils import*
import numpy as np
import os
from pprint import pprint

langs = ['en', 'eu', 'ca', 'gl', 'it', 'fr', 'nl', 'de', 'da', 'no', 'sv']
f1s = []

# english is a bit different
_, _, test_data = open_dataset(os.path.join('annotation', 'datasets', 'en'), 'en')
l, r, t, y = zip(*test_data)
counts = np.bincount(y)
b = np.argmax(counts)
pred = [b] * len(y)
f1 = per_class_f1(y, pred).mean()
f1s.append(f1)

# other languages
for lang in langs[1:]:
    test_data = open_dataset(
                    os.path.join('annotation', 'datasets',lang), lang)
    l, r, t, y = zip(*test_data)
    counts = np.bincount(y)
    b = np.argmax(counts)
    pred = [b] * len(y)
    f1 = per_class_f1(y, pred).mean()
    f1s.append(f1)


print('{0}    {1}    {2}    {3}    {4}    {5}    {6}    {7}    {8}    {9}    {10}'.format(*langs))  
print('{0:.2}  {1:.2}  {2:.2}  {3:.2}  {4:.2}  {5:.2}  {6:.2}  {7:.2}  {8:.2}  {9:.2}  {10:.2}'.format(*f1s))
