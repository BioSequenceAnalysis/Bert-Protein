# 开始使用

克隆本仓库。

```
git clone git@github.com:BioSequenceAnalysis/Bert-Protein.git
```

从下面地址下载本项目所使用的模型。

[百度网盘（提取密码：a1yi）](https://pan.baidu.com/s/1RH_OeHkpzMFa-YBnTMHerA)

模型下载完成后放置于根目录之下。

# 预训练

创建预训练所需数据。

```
sh create_data.sh
```

在开始预训练之前，需要确认一些内容。

> input_file 为以`tfrecord`形式保存的预训练数据。\
> output_dir 为模型输出目录。\
> bert_config_file 定义了模型结构。\
> train_batch_size 由你的机器性能决定。\
> num_train_steps 自行调整。

更多参数参考[Bert 官方文档](https://github.com/google-research/bert)。

# 微调、评估及模型保存

准备好预训练模型，开始执行微调操作，应运行以下代码

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

每个参数的含义如下，您应根据需要对参数进行修改。您也可以打开文件`ljy_run_classifier.py`并更改第16-32行中的代码以修改这些参数的默认值。

>do_eval：训练后是否评估模型\
>do_save_model：训练后是否保存模型\
>data_name：训练集的名称\
>batch_size：批量大小\
>num_train_epochs：训练epoch数目\
>warmup_proportion：预热比例\
>learning_rate：学习率\
>using_tpu：是否使用TPU\
>seq_length：序列长度\
>data_root：要使用的训练集的位置\
>vocab_file：字典的位置\
>init_checkpoint：模型的初始化节点\
>bert_config：BERT配置文件\
>save_path：将训练好的模型的保存位置

# 进行预测

您可以通过命令预测蛋白质数据

```python ljy_predict_AMP.py```

 您应该根据需要更改`ljy_predict_AMP.py`中第167-172的代码。

> data_name：testing集的位置\
> out_file：测试结果的存储位置\
> model_path：训练模型的位置\
> step：分词\
> config_file：BERT配置\
> vocab_file：字典的位置
