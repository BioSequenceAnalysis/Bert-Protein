# 设置可以使用的GPU
export CUDA_VISIBLE_DEVICES=1
NUM = '1'
DATANAME = "all_data"
python ./ljy_run_classifier.py \
  --dataname=$DATANAME \
  --do_eval=True \
  --learning_rate=2e-6 \
  --data_root='./dataset/${NUM}kmer_tfrecord/${DATANAME}/' \
  --vocab_file='./vocab/vocab_${NUM}kmer.txt' \
  --init_checkpoint='./model/${NUM}kmer_model/model.ckpt' \
  --bert_config='./bert_config_${NUM}.json' \
  --save_path='./model/classifier_model_${NUM}kmer_${DATANAME}./model.ckpt'