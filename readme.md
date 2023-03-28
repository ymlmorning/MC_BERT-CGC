This is the optimized code based on the paper [An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition](https://arxiv.org/abs/2208.04534)

Chinese electronic medical record data set is stored in the directory preprocess\outputs\ace2004, we do not share private data.

```shell
python train_new_bert.py -n 3 --lr 1e-4 --cnn_dim 160 --biaffine_size 200 --n_head 5 -b 5 -d ace2004 --logit_drop 0 --cnn_depth 1 |tee result_bert//result1.txt
```

Here, we set `n_heads`, `cnn_dim` and `biaffine_size` for small number of parameters, based on our experiment, reduce `n_head` and
enlarge `cnn_dim` and `biaffine_size` should get slightly better performance.

### Customized data
If you want to use your own data, please organize your data line like the following way, the data folder should 
have the following files
```text
customized_data/
    - train.jsonelines
    - dev.jsonlines
    - test.jsonlines
```
in each file, each line should be a json object, like the following
```text
{"tokens": ["'个人", "史", "':'", "出生", "并", "久", "居于", "本地", "否认", "疫水", "疫区", "接触", "史", "否认", "其他", "放射性", "物质", "及", "毒物", "接触", "史", ""], "doc_id": "", "sent_id": "", "entity_mentions": [{"start": "1", "end": "1", "entity_type": "clinical_manifestation", "text": "史"}, {"start": "3", "end": "6", "entity_type": "clinical_manifestation", "text": "出生并久居于"}, {"start": "4", "end": "7", "entity_type": "clinical_manifestation", "text": "并久居于本地"}, {"start": "5", "end": "7", "entity_type": "clinical_manifestation", "text": "久居于本地"}, {"start": "9", "end": "9", "entity_type": "position", "text": "疫水"}, {"start": "10", "end": "10", "entity_type": "position", "text": "疫区"}, {"start": "10", "end": "12", "entity_type": "clinical_manifestation", "text": "疫区接触史"}, {"start": "11", "end": "11", "entity_type": "operation", "text": "接触"}, {"start": "11", "end": "12", "entity_type": "clinical_manifestation", "text": "接触史"}, {"start": "12", "end": "12", "entity_type": "clinical_manifestation", "text": "史"}, {"start": "14", "end": "14", "entity_type": "clinical_manifestation", "text": "其他"}, {"start": "14", "end": "16", "entity_type": "clinical_manifestation", "text": "其他放射性物质"}]}
```
the entity `start` and `end` is inclusive and exclusive, respectively.
