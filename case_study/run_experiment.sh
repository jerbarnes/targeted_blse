#!/bin/bash

# add paths to embeddings here
emb_dir=../embeddings/mono
muse_dir=../embeddings/muse

echo "Binary"
for lang in eu ca gl it fr nl de no sv da; do
	for binary in True False; do
		for training_data in twitter usage semeval; do
 			echo training English on "$training_data" and testing on "$lang" - binary="$binary"

 			# BLSE
 			python aspect_BLSE.py -tl "$lang" -te "$emb_dir"/sg-300-"$lang".txt -bi "$binary" --src_dataset "$training_data"

 			# Vecmap
 			python aspect_vecmap.py -tl "$lang" -te "$emb_dir"/sg-300-"$lang".txt -bi "$binary" --src_dataset "$training_data"

 			# MUSE
			python aspect_muse.py -tl "$lang" -se "$muse_dir"/en-"$lang"/vectors-en.txt -te "$muse_dir"/en-"$lang"/vectors-"$lang".txt -bi "$binary"  --src_dataset "$training_data"

			# MT
			python aspect_MT.py -tl "$lang" -se "$emb_dir"/sg-300-en.txt -bi "$binary"  --src_dataset "$training_data"
		done;
	done;
done;