# 简介
该项目为基于Bert所训练出的抗菌肽识别模型。模型使用大量从UmiPort数据库下载的蛋白质序列数据预训练。在四个不同的数据集上进行训练并对其评估，最终合并训练结果。

# 开始使用
克隆本仓库。
```
git clone git@github.com:BioSequenceAnalysis/Bert-Protein.git
```
从下面地址下载本项目所使用的模型。

[谷歌云](https://drive.google.com/open?id=1VSi-bdPpT0Z1ytmhVxbHGGjZDtQNLjm6)

[百度网盘（提取密码：nxy5）](https://pan.baidu.com/s/1y2aNlHWiAckNkPVugpEwUA)

模型下载完成后放置于根目录之下。

# 预训练
创建预训练所需数据。

```
sh create_data.sh
```

在开始预训练之前，需要确认一些内容。

> input_file 为以`tfrecord`形式保存的预训练数据。
> output_dir 为模型输出目录。
> bert_config_file 定义了模型结构。
> train_batch_size 由你的机器性能决定。
> num_train_steps 自行调整。

更多参数参考[Bert 官方文档](https://github.com/google-research/bert)。

# 微调、评估及模型保存
开始微调，具体参数参考`ljy_run_classifier.py`内设置。
```
sh run_fine_tune.sh
```

# 进行预测
```
python ljy_predict_AMP.py f1 f2
```
f1为输入文件，f2为输出文件。