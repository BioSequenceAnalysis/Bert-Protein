python create_pretraining_data.py \
	--input_file=./pre_train_data/uniprot_data.txt \
	--output_file=./pre_train_data/uniprot_data.tfrecord \
	--vocab_file=./vocab/vocab_1kmer.txt \
	--do_lower_case=True \
	--max_seq_length=128 \
	--max_predictions_per_seq=20 \
	--masked_lm_prob=0.15 \
	--random_seed=12345 \
	--dupe_factor=5
