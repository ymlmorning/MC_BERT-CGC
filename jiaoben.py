import os
os.system("python train_new_bert.py -n 3 --lr 1e-4 --cnn_dim 160 --biaffine_size 200 --n_head 5 -b 5 -d ace2004 --logit_drop 0 --cnn_depth 1 |tee result_bert//result1.txt")
os.system("python train_new_bert.py -n 3 --lr 1e-4 --cnn_dim 160 --biaffine_size 200 --n_head 5 -b 5 -d ace2004 --logit_drop 0 --cnn_depth 1 |tee result_bert//result2.txt")
os.system("python train_new_bert.py -n 3 --lr 1e-4 --cnn_dim 160 --biaffine_size 200 --n_head 5 -b 5 -d ace2004 --logit_drop 0 --cnn_depth 1 |tee result_bert//result3.txt")
os.system("python train_new_bert.py -n 3 --lr 1e-4 --cnn_dim 160 --biaffine_size 200 --n_head 5 -b 5 -d ace2004 --logit_drop 0 --cnn_depth 1 |tee result_bert//result4.txt")
os.system("python train_new_bert.py -n 3 --lr 1e-4 --cnn_dim 160 --biaffine_size 200 --n_head 5 -b 5 -d ace2004 --logit_drop 0 --cnn_depth 1 |tee result_bert//result5.txt")