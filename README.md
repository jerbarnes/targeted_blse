Targeted Cross-lingual Sentiment Analysis via Embedding Projection
==============

This repository hosts the source code and data for the work below:


Requirements to run the experiments
--------
- Python 3
- NumPy
- sklearn [http://scikit-learn.org/stable/]
- pytorch [http://pytorch.org/]



Usage
--------

First, clone the repo:

```
git clone https://github.com/jbarnesspain/targeted_blse
cd targeted_blse
```


Then, get monolingual embeddings, either by training your own,
or by downloading the [pretrained embeddings](https://drive.google.com/open?id=1GpyF2h0j8K5TKT7y7Aj0OyPgpFc8pMNS) mentioned in the paper,
unzipping them and putting them in the 'embeddings' directory:


You can then cd into the experiment_2 and casestudy directory and
run either experiment using the run_experiment.sh script:

```
cd experiment_2
./run_experiment.sh
```


License
-------

Copyright (C) 2018, Jeremy Barnes

Licensed under the terms of the Creative Commons CC-BY public license
