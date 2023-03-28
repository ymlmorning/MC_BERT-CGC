import os
for i in range(10,20):
    for j in range(0,5):
       if j!=0:continue
       ss="python train_new"+str(j)+".py -n 10 --lr 1e-4 --cnn_dim 160 --biaffine_size 200 --n_head 5 -b 5 -d ace2004 --logit_drop 0 --cnn_depth 1 --seed 500|tee result_bert"+str(j)+"//result"+str(i)+".txt"
       os.system(ss)
