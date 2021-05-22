[中文文档](https://github.com/JianyuanLin/Bert-Protein/blob/master/README_ZH.md)

# Introduction

This is a model for antimicrobial peptides recognition based on BERT which is proposed.\
We pre-train a BERT model through amount of proteins sequences downloaded from UniPort.\
We fine-tune the model on 4 datasets and evaluate its performance.\
We merge all the datasets and train a comprehensive prediction model.

# How to Start

You should first clone the project by command

```
git clone git@github.com:BioSequenceAnalysis/Bert-Protein.git
```

Then you need to download models and datasets from the address:

>https://pan.baidu.com/s/1RH_OeHkpzMFa-YBnTMHerA

The password is a1yi

Then you should unzip these zips and put them on the root of the project.
 # Pre-training

 ## Data Format
Before you create data, you should make sure the input file is properly formatted(just like this).
```
train 0   SEQUENCE
```
We also provide a sample code(fasta2tsv.py) for using the transformation format (which may require you to change something to suit your needs).
 ## Create tfrecord file

 You should create data for pre-train by the command

```sh create_data.sh```

You should ensure the content of file pre_train.sh
>input_file is your input data for pre-training whose format is tf_record.\
>output_dir is the dir of your output model.\
>bert_config_file defines the structure of the model.\
>train_batch_size should be change more little if your computer don't support so big batch size.\
>You can change the num_train_steps by yourself.

After ensuring the content, then you can pre-trian your model by the command:

```sh pre_train.sh```

 # Fine-Tuning & Evaluation & Save Model
 When you ready to fine-tune the model or do other, you should run the following code

 ```
 python ljy_run_classifier.py \
 --do_eval False \
 --do_save_model True \
 --data_name AMPScan \
 --batch_size 32 \
 --num_train_epochs 50 \
 --warmup_proportion 0.1 \
 --learning_rate 2e-5 \
 --using_tpu False \
 --seq_length 128 \
 --data_root ./dataset/1kmer_tfrecord/AMPScan/ \
 --vocab_file ./vocab/vocab_1kmer.txt \
 --init_checkpoint ./model/1kmer_model/model.ckpt \
 --bert_config ./bert_config_1.json \
 --save_path ./model/AMPScan_1kmer_model/model.ckpt
 ```

The meaning of each parameter is as follows, you should change these according to your needs. You can also open file ljy_run_classifier and change the codes in row 16-32 to modify the default values of these parameters.

> do_eval: whether to evaluate the model after training\
> do_save_model: whether to save the model after training\
> data_name: the name of the training set\
> batch_size: batch size\
> num_train_epochs: training epochs\
> warmup_proportion: proportion of warmup\
> learning_rate: learning rate\
> using_tpu: Whether to use TPU\
> seq_length: sequence length\
> data_root: the location of the training set to be used\
> vocab_file: location of dictionary\
> init_checkpoint: initialization node of the model\
> bert_config: BERT configuration\
> save_path: where to save the trained model


 # Predict

You can predict your proteins data by command

```python ljy_predict_AMP.py```

 You should change the codes in row 167-172 according to your needs.

> data_name: location of the testing set\
> out_file: storage location of test results\
> model_path: the location of the trained model\
> step: word segmentation\
> config_file: BERT configuration\
> vocab_file: location of dictionary
