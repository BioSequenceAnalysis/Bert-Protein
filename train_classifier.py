# coding:utf-8
from run_classifier import DataProcessor, InputExample, \
    file_based_input_fn_builder, file_based_convert_examples_to_features, create_model
from my_model_fn import model_fn_builder
import tensorflow as tf
import os
import tokenization
import modeling


class Flags:
    def __init__(self):
        self.do_lower_case = True
        self.init_checkpoint = "./bert_classifier/model.ckpt-10000000"
        self.bert_config_file = "./bert_config.json"
        self.output_dir = "./bert_classifier/output/"
        self.vocab_file = "./vocab/vocab.txt"
        self.save_checkpoints_steps = 1000
        self.iterations_per_loop = 1000
        self.data_dir = "./bert_classifier/tsv_data/"
        self.train_batch_size = 32
        self.eval_batch_size = 8
        self.predict_batch_size = 8
        self.num_train_epochs = 3.0
        self.warmup_proportion = 0.1
        self.learning_rate = 2e-5
        self.max_seq_length = 128
        self.watch_times = 1


class MyProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[3])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def main():
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    processor = MyProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=8,   # 这个参数在没有调用到tpu的时候实际上无法用到
            per_host_input_for_training=is_per_host))

    train_examples = processor.get_train_examples(FLAGS.data_dir)
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    # 这里的train_examples在前面的processor中已经处理出来了
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    eval_steps = None
    eval_drop_remainder = False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    for i in range(FLAGS.watch_times):
        num_train_steps = int(
            (i + 1) * len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_one_hot_embeddings=False)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": 8})

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "a") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                writer.write("%s" % "The" + str(i) + "-th eval\n")
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write("\n")

main()
