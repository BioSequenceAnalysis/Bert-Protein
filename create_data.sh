python create_pretraining_data.py \
	 --input_file=./uniprot_data/all_protein_3kmer.fasta \
	    --output_file=./record/all_protein_3kmer.tfrecord \
	    --vocab_file=./vocab/vocab.txt \
	        --do_lower_case=True \
		  --max_seq_length=128 \
		    --max_predictions_per_seq=20 \
		      --masked_lm_prob=0.15 \
		        --random_seed=12345 \
			  --dupe_factor=5
