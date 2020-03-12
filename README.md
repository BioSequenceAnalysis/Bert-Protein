# Introduction
This is a model for antimicrobial peptides recognition based on BERT which is proposed.
We pretrained a BERT model through amount of proteins sequences downloaded from UniPort.
We train the model on 4 datasets and evaluate its performance.
We merge all the datasets and train a model.
# How to Start
You should first clone the project by command
>git clone https://github.com/JianyuanLin/Bert-Protein

Then you need to download models and datasets from the address:
>https://drive.google.com/open?id=1VSi-bdPpT0Z1ytmhVxbHGGjZDtQNLjm6

or the address:

>https://pan.baidu.com/s/1y2aNlHWiAckNkPVugpEwUA

The password is nxy5
 
Then you should uzip these zips and put them on the root of the project.
 # Pre-training
 
 You should create data for pre-train by the command
 >sh create_data.sh
 
You should ensure the content of file pre_train.sh
>input_file is your input data for pre-training whose format is tf_record.  
output_dir is the dir of your output model.  
bert_config_file defines the structure of the model.
train_batch_size should be change more little if your computer don't support so big batch size.
You can change the num_train_steps by yourself.

After ensuring the content, then you can pre-trian your model by the command:
>sh pre_train.sh

 # Fine-Tuning & Evaluation & Save Model
 When you ready to fine-tune the model or do other, you should open file ljy_run_classifier first.
 You should change the codes in row 31-62 according to your needs.
> do_eval and do_save are used to indicate if you want to evaluate the model or save the final model.  
If the do_save is True then the final model will be saved in path "./model/classifier_model/"
train_dict and test_dict record the numbers of samples in training sets and test sets.  
init_chechpoint is the model which is used to train.

 
 # Predict
You can predict your proteins data by command
>python ljy_predict_AMP.py f1 f2  


f1 is the fasta format file contains your proteins data and f2 is the output file.
