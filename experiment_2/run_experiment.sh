#!/bin/bash

# add path to embeddings here
emb_dir=../embeddings/


# OpeNER experiment
# best hyperparameters for OpeNER here
lr=0.003
wd=3e-05
bs=100
alpha=0.5
epochs=50

for lang in es ca eu; do
	for bi in True False; do
		# BLSE
		python3 aspect_BLSE.py -tl "$lang" -src_da opener -trg_da opener -se "emb_dir"/sg-300-en.txt -te "emb_dir"/sg-300-"$lang".txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"

		# VecMap
		python3 aspect_vecmap.py -tl "$lang" -src_da opener -trg_da opener -se "emb_dir"/sg-300-en.txt -te "emb_dir"/sg-300-"$lang".txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"

		# Barista
		python3 aspect_barista.py -tl "$lang" -src_da opener -trg_da opener -se "emb_dir"/barista/sg-300-window4-negative20_en_"$lang".txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"

		# MUSE
		python3 aspect_MUSE.py -tl "$lang" -src_da opener -trg_da opener -se "emb_dir"/muse/en-"$lang"/vectors-en.txt -te "emb_dir"/muse/en-"$lang"/vectors-"$lang".txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"

		# MT
		python3 aspect_MT.py -tl "$lang" -src_da opener -trg_da opener -se "emb_dir"/sg-300-en.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"
	done;
done;


# SemEval experiment
# best hyperparameters for SemEval here
lr=0.003
wd=3e-05
bs=100
alpha=0.5
epochs=100

for bi in True False; do
	# BLSE
	python3 aspect_BLSE.py -tl es -src_da semeval -trg_da semeval -se "emb_dir"/sg-300-en.txt -te "emb_dir"/sg-300-es.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"

	# VecMap
	python3 aspect_vecmap.py -tl es -src_da semeval -trg_da semeval -se "emb_dir"/sg-300-en.txt -te "emb_dir"/sg-300-es.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"


	# Barista
	python3 aspect_barista.py -tl es -src_da semeval -trg_da semeval -se "emb_dir"/barista/sg-300-window4-negative20_en_es.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"

	# MUSE
	python3 aspect_MUSE.py -tl es -src_da semeval-trg_da semeval -se "emb_dir"/muse/en-es/vectors-en.txt -te "emb_dir"/muse/en-es/vectors-es.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"

	# MT
	python3 aspect_MT.py -tl es -src_da opener -trg_da opener -se "emb_dir"/sg-300-en.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"
done;


# USAGE experiment
# best hyperparameters for USAGE here
lr=0.003
wd=3e-05
bs=100
alpha=0.5
epochs=50

for bi in True False; do
	# BLSE
	python3 aspect_BLSE.py -tl de -src_da usage -trg_da usage -se "emb_dir"/sg-300-en.txt -te "emb_dir"/sg-300-de.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"

	# VecMap
	python3 aspect_vecmap.py -tl de -src_da usage -trg_da usage -se "emb_dir"/sg-300-en.txt -te "emb_dir"/sg-300-de.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"


	# Barista
	python3 aspect_barista.py -tl de -src_da usage -trg_da usage -se "emb_dir"/barista/sg-300-window4-negative20_en_de.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"

	# MUSE
	python3 aspect_MUSE.py -tl de -src_da usage -trg_da usage -se "emb_dir"/muse/en-de/vectors-de.txt -te "emb_dir"/muse/en-de/vectors-de.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"

	# MT
	python3 aspect_MT.py -tl de -src_da usage -trg_da usage -se "emb_dir"/sg-300-en.txt -lr "$lr" -wd "$wd" -bs "$bs" -a "$alpha" -e "$epochs" -bi "$bi"
done;

