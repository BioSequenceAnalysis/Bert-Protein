# coding:utf-8

from run_classifier import ColaProcessor
import tokenization
from run_classifier import file_based_convert_examples_to_features
import random


def rand_data(input_file, out_file):
    with open(input_file) as f:
        lines = f.readlines()
    random.shuffle(lines)
    print(lines)
    with open(out_file, "w") as f:
        for line in lines:
            f.write(line)


def create_tfrecord(kmer, data_name):
    tsv_root = "dataset/" + str(kmer) + "kmer_tsv_data/" + data_name
    tfrecord_root = "dataset/" + str(kmer) + "kmer_tfrecord/" + data_name
    rand_data(tsv_root + "/sorted_tr.tsv",
              tsv_root + "/train.tsv",)
    rand_data(tsv_root + "/sorted_te.tsv",
              tsv_root + "/dev.tsv", )
    vocab_file = "vocab/vocab_" + str(kmer) + "kmer.txt"
    processor = ColaProcessor()
    label_list = processor.get_labels()
    examples = processor.get_dev_examples(tsv_root + "/")
    train_file = tfrecord_root + "/dev.tf_record"
    tokenizer = tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
            examples, label_list, 128, tokenizer, train_file)

    examples = processor.get_train_examples(tsv_root + "/")
    train_file = tfrecord_root + "/train.tf_record"
    tokenizer = tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
            examples, label_list, 128, tokenizer, train_file)


for kmer in range(1, 5):
    # for data_name in ["AMPScan", "Bi-LSTM", "iAMP", "MAMP"]:
    for data_name in ["all_data"]:
        create_tfrecord(kmer, data_name)
