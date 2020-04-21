# coding:utf-8
import modeling
import tokenization
from run_classifier import create_model, file_based_input_fn_builder
import tensorflow as tf
import optimization
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, \
    confusion_matrix, roc_curve
import os
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('do_eval', False, '是否在训练后评估')
tf.app.flags.DEFINE_boolean('do_save_model', True, '是否在训练后保存模型')
tf.app.flags.DEFINE_string('data_name', 'all_data', '导入的数据名')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch大小')
tf.app.flags.DEFINE_integer('num_train_epochs', 50, '训练的轮次')
tf.app.flags.DEFINE_float('warmup_proportion', 0.1, '预热的训练比例')
tf.app.flags.DEFINE_float('learning_rate', 2e-5, '学习率')
tf.app.flags.DEFINE_boolean('using_tpu', False, '是否使用TPU')
tf.app.flags.DEFINE_float('seq_length', 128, '序列长度')
tf.app.flags.DEFINE_string(
    'data_root', './dataset/1kmer_tfrecord/all_data/', '使用数据集的根目录')
tf.app.flags.DEFINE_string('vocab_file', './vocab/vocab_1kmer.txt', '词典目录')
tf.app.flags.DEFINE_string(
    'init_checkpoint', "./model/1kmer_model/model.ckpt", '模型的初始化节点')
tf.app.flags.DEFINE_string('bert_config', "./bert_config_1.json", 'bert配置')
tf.app.flags.DEFINE_string(
    'save_path', "./model/classifier_model/model.ckpt", '模型保存位置')


def count_trues(pre_labels, true_labels):
    shape = true_labels.shape
    zeros = np.zeros(shape=shape)
    ones = np.ones(shape=shape)
    pos_example_index = (true_labels == ones)
    neg_example_index = (true_labels == zeros)
    right_example_index = (pre_labels == true_labels)
    true_pos_examples = np.sum(np.logical_and(
        pos_example_index, right_example_index))
    true_neg_examples = np.sum(np.logical_and(
        neg_example_index, right_example_index))
    return np.sum(pos_example_index), np.sum(neg_example_index), true_pos_examples, true_neg_examples


def main():
    # 以下是输入的参数，当更换词典时，请修改文件 bert_config.json中的vocab_size的值
    do_eval = FLAGS.do_eval     # 是否在训练之后进行评估
    do_save_model = FLAGS.do_save_model    # 是否存储训练的模型
    data_name = FLAGS.data_name  # 选定数据名，用于导入对应的路径的数据
    train_dict = {"AMPScan": 2132,
                  "BiLSTM": 4174,
                  "iAMP": 1758,
                  "MAMPs": 5234,
                  "fold": 2928,
                  'Legionellapneumophilatmp': 998,
                  "all_data": 8978,
                  }     # 记录了各个训练集的样本数量
    test_dict = {"AMPScan": 1424,
                 "BiLSTM": 1156,
                 "iAMP": 1839,
                 "MAMPs": 1666,
                 "fold": 2119,
                 'Legionellapneumophilatmp': 128,
                 "all_data": 8978,
                 }  # 记录了各个测试集的样本数量
    tf.logging.set_verbosity(tf.logging.INFO)
    train_example_num = train_dict[data_name]   # 获取训练集样本数量
    test_example_num = test_dict[data_name]     # 获取测试集样本数量
    batch_size = FLAGS.batch_size  # batch的大小，如果gpu显存不够，可以考虑减小一下，尽量2的批次
    # 根据batch的大小计算出训练集的batch的数量
    train_batch_num = math.ceil(train_example_num / batch_size)
    test_batch_num = math.ceil(
        test_example_num / batch_size)       # 计算出测试集的batch的数量
    num_train_epochs = FLAGS.num_train_epochs   # 训练的次数，代表过几遍训练集
    warmup_proportion = FLAGS.warmup_proportion  # 预热的训练比例，保持不变即可
    learning_rate = FLAGS.learning_rate    # 学习率
    use_tpu = FLAGS.using_tpu         # 无视好了
    seq_length = FLAGS.seq_length        # 这边默认设定128，不需要更改
    data_root = FLAGS.data_root    # 根据实际调用的数据集来这里修改数据路径
    vocab_file = FLAGS.vocab_file             # 词典
    init_checkpoint = FLAGS.init_checkpoint  # 模型初始化检查点，记录了模型的权重
    bert_config = modeling.BertConfig.from_json_file(
        FLAGS.bert_config)    # 根据该文件可以配置模型的结构

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75   # 这三行用来防止直接占用全GPU
    # 输入训练集，这个文件是采用ljy_tsv2record生成的
    input_file = data_root + "/train.tf_record"
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    num_train_steps = int(
        train_example_num / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)  # 配置训练次数
    input_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    input_mask = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    segment_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    label_ids = tf.placeholder(
        dtype=tf.int32, shape=(None,))   # 留四个占位符，用以输入数据和标签
    is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
    is_training = True
    num_labels = 2  # 二分类，共两个标签
    use_one_hot_embeddings = False
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)  # 该函数生成BERT模型的计算图，并返回计算图的总损失、各样本损失、两个输出
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None

    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        # 到这里为止根据检查点初始化计算图变量，相当于加载模型
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)  # 初始化优化器，用于梯度下降使用
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }   # 生成输入数据所需的变量
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
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,))
        return d

    train_data = input_fn({"batch_size": batch_size})   # 生成训练集
    iterator = train_data.make_one_shot_iterator().get_next()   # 生成训练集数据迭代器，迭代器会在循环中输出数据
    if do_eval:
        input_file = data_root + "/dev.tf_record"
        dev_data = input_fn({"batch_size": batch_size})
        dev_iterator = dev_data.make_one_shot_iterator().get_next()  # 生成验证机迭代器
    val_accs = []
    sps = []
    sns = []
    if do_save_model:
        saver = tf.train.Saver()    # 生成存储节点
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)  # 初始化计算图

        for step in range(num_train_epochs):
            start_time = time.time()
            for _ in range(train_batch_num):
                examples = sess.run(iterator)  # 运行迭代器生成样本
                # print(examples)
                _, loss = \
                    sess.run([train_op, total_loss],
                             feed_dict={input_ids: examples["input_ids"],
                                        input_mask: examples["input_mask"],
                                        segment_ids: examples["segment_ids"],
                                        label_ids: examples["label_ids"]})  # 进行梯度下降，并取回loss值，喂入迭代器生成的数据
            print("step:", step, " loss:", round(loss, 4), end=" ")
            all_prob = []
            all_labels = []
            all_pre_labels = []
            if not do_eval:
                end_time = time.time()
                eta_time = (end_time - start_time) * \
                    (num_train_epochs - step - 1)
                print(" eta time:", eta_time, "s")
                continue
            for _ in range(test_batch_num):
                examples = sess.run(dev_iterator)
                loss, prob = \
                    sess.run([total_loss, probabilities],
                             feed_dict={input_ids: examples["input_ids"],
                                        input_mask: examples["input_mask"],
                                        segment_ids: examples["segment_ids"],
                                        label_ids: examples["label_ids"]})
                all_prob.extend(prob[:, 1].tolist())
                all_labels.extend(examples["label_ids"].tolist())
                pre_labels = np.argmax(prob, axis=-1).tolist()
                all_pre_labels.extend(pre_labels)
            acc = accuracy_score(all_labels, all_pre_labels)
            val_accs.append(acc)
            auc = roc_auc_score(all_labels, all_prob)
            mcc = matthews_corrcoef(all_labels, all_pre_labels)
            c_mat = confusion_matrix(all_labels, all_pre_labels)
            sn = c_mat[1, 1] / np.sum(c_mat[1, :])    # 预测正确的正样本
            sp = c_mat[0, 0] / np.sum(c_mat[0, :])    # 预测正确的负样本
            sps.append(sp)
            sns.append(sn)  # 计算各项性能指标
            end_time = time.time()
            eta_time = (end_time - start_time) * (num_train_epochs - step - 1)
            print("SE:", sn, " SP:", sp, " ACC:", acc, " MCC:", mcc, " auROC:", auc, " eta time:",
                  eta_time, "s")

        if do_save_model:
            save_path = saver.save(
                sess, FLAGS.save_path)  # 存储模型


main()
