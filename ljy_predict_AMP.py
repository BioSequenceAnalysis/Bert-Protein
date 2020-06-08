# CODing:utf-8

import modeling
import tokenization
from run_classifier import create_model, file_based_input_fn_builder, ColaProcessor, \
    file_based_convert_examples_to_features
import tensorflow as tf
import optimization
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, \
    confusion_matrix, roc_curve
import os
import time
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def fasta2record(input_file, output_file, vocab_file, step=1):
    # This function gets an input_file which is .fasta
    # This function returns the numbers of sequences in input_file
    # This function will check if the input_file is right
    with open(input_file) as f:
        lines = f.readlines()
    print(lines)
    for index, line in enumerate(lines):
        print(line)
        if index % 2 == 0:
            if line[0] != ">":
                print("Row " + str(index + 1) + " is wrong!")
                exit()
        else:
            if line[0] == ">":
                print("Row " + str(index + 1) + " is wrong!")
                exit()
    seq_num = int(len(lines) / 2)
    with open("temp.tsv", "w") as f:
        for line in lines:
            if line[0] != ">":
                seq = ""
                line= line.strip()
                length = len(line)
                # step = 1
                for i in range(0, length, step):
                    if length - i >= step:
                        seq += line[i:i+step] + " "
                    else:
                        seq += line[i:] + " "
                seq += "\n"
                f.write("train\t1\t\t" + seq)
    processor = ColaProcessor()
    label_list = processor.get_labels()
    examples = processor.ljy_get_dev_examples("temp.tsv")
    train_file = "predict.tf_record"
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
        examples, label_list, 128, tokenizer, train_file)
    return seq_num


def main(data_name, out_file, model_path, step=1, config_file="./bert_config_1.json",
         vocab_file="./vocab/vocab_1kmer.txt"):
    tf.logging.set_verbosity(tf.logging.INFO)
    batch_size = 32
    use_tpu = False
    seq_length = 128
    # vocab_file = "./vocab/vocab_2kmer.txt"
    init_checkpoint = model_path
    bert_config = modeling.BertConfig.from_json_file(config_file)
    learning_rate = 2e-5
    num_train_steps = 100
    num_warmup_steps = 10

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    samples_num = fasta2record(data_name, "predict.tf_record", vocab_file, step=step)
    batch_num = math.ceil(samples_num / batch_size)
    input_file = "predict.tf_record"
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    input_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    input_mask = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    segment_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    label_ids = tf.placeholder(dtype=tf.int32, shape=(None,))
    is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
    num_labels = 2
    use_one_hot_embeddings = False
    is_training = True
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None

    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    drop_remainder = False

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,))
        return d

    predict_data = input_fn({"batch_size": batch_size})
    iterator = predict_data.make_one_shot_iterator().get_next()
    all_prob = []
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for _ in range(batch_num):
            examples = sess.run(iterator)
            prob = \
                sess.run(probabilities,
                         feed_dict={input_ids: examples["input_ids"],
                                    input_mask: examples["input_mask"],
                                    segment_ids: examples["segment_ids"],
                                    label_ids: examples["label_ids"]})
            all_prob.extend(prob[:, 1].tolist())
    # print(all_prob)
    # print(len(all_prob))
    with open(data_name) as f:
        lines = f.readlines()
    with open(out_file, "w") as f:
        index = 0
        for line in lines:
            if line[0] == ">":
                f.write(line)
            else:
                f.write(line.strip() + " " + str(all_prob[index]) + "\n")
                index += 1


if __name__ == '__main__':
    # fold_num = "fold4"
    # cate = "NAMP"
    # main("dataset/Ori_dataset/" + fold_num + "/" + cate + "_te.fa",
    #      "pre_result/" + fold_num + "_" + cate + "_result.txt",
    #      "model/" + fold_num + "/model.ckpt")
    main(data_name="dataset/Ori_dataset/AMPScan/NAMP_te.fa",
         out_file="pre_result/AMPScan_NAMP_data_result.txt",
         model_path="model/classifier_model/model2.ckpt",
         step=2,
         config_file="./bert_config_2.json",
         vocab_file="./vocab/vocab_2kmer.txt")
