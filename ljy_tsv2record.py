# coding:utf-8

from run_classifier import ColaProcessor
import tokenization
from run_classifier import file_based_convert_examples_to_features
import random


data_name = "AMPScan"
input_root = "./pre_model_1kmer_600w/tsv_data/"
output_root = "./pre_model_1kmer_600w/tfrecord/"
vocab_file = "./vocab/vocab_1kmer.txt"


def rand_data(input_file, out_file):
    with open(input_file) as f:
        lines = f.readlines()
    random.shuffle(lines)
    print(lines)
    with open(out_file, "w") as f:
        for line in lines:
            f.write(line)


rand_data(input_root + data_name + "/sorted_train.tsv",
          input_root + data_name + "/train.tsv")

rand_data(input_root + data_name + "/sorted_dev.tsv",
          input_root + data_name + "/dev.tsv")


processor = ColaProcessor()
label_list = processor.get_labels()
examples = processor.get_dev_examples(input_root + data_name + "/")
train_file = output_root + data_name + "/dev.tf_record"
tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=True)
file_based_convert_examples_to_features(
        examples, label_list, 128, tokenizer, train_file)

examples = processor.get_train_examples(input_root + data_name + "/")
train_file = output_root + data_name + "/train.tf_record"
tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=True)
file_based_convert_examples_to_features(
        examples, label_list, 128, tokenizer, train_file)
