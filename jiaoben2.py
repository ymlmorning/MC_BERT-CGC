import os
for i in range(10,20):
    for j in range(3,4):
        ss="python train_new"+str(j)+".py -n 3 --lr 1e-4 --cnn_dim 160 --biaffine_size 200 --n_head 5 -b 5 -d ace2004 --logit_drop 0 --cnn_depth 1 |tee result_bert"+str(j)+"//result"+str(i)+".txt"
        os.system(ss)
