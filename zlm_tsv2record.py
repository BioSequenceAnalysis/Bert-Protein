# coding:utf-8

from run_classifier import ColaProcessor
import tokenization
from run_classifier import file_based_convert_examples_to_features

data_name = "Legionellapneumophilatmp"
input_root = "./dataset/1kmer_tsv_data/"
output_root = "./dataset/1kmer_tfrecord/"
vocab_file = "./vocab/vocab_1kmer.txt"

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
