#!/usr/bin/env python3

import json
import pickle
import os, sys

vocab_path = '/data/CoCo/processed_restricted/vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

idx2obj = vocab.get('idx2obj')
idx2pred = vocab.get('idx2pred')
pred2idx = vocab.get('pred2idx')
num_objects = max(idx2obj.keys())
num_predicates = len(pred2idx)

def append_sg(path, sg_dict):
    with open(path, 'rb') as f:
        batch = pickle.load(f)
    for iid in batch:
        objs = batch[iid]['objs']
        triples = batch[iid]['triples']
        objects = [idx2obj[i] for i in objs]
        rels = [[i[0], idx2pred[i[1]], i[2]] for i in triples]
        sg_dict[iid] = {'objects':objects, 'relationships':rels}

if(__name__=='__main__'):

    sg_dict = {}
    batch_dir = '/data/CoCo/val_processed_restricted/batch'
    for root, dirs, files in os.walk(batch_dir):
        for f in files:
            if(f.endswith('.pkl')):
                append_sg(os.path.join(root,f),sg_dict)
    
    key_list = list(sg_dict.keys())
    key_list.sort()
    json_output = []
    r = int(sys.argv[1])
    start = 500*r; end = 500*(r+1)
    count = 0
    print('key_list',len(key_list))
    print('sg_dict',len(sg_dict))
    #print(key_list)
    for key in key_list:
        assert len(sg_dict[key]['objects'])>2
        assert len(sg_dict[key]['relationships'])>2
        if(count>=start):
            json_output.append(sg_dict[key])
        count += 1
        if(count==end):
            break
    print('output:',len(json_output))
    with open('val_sg.json','w') as f:
        json.dump(json_output, f)
    print('json saved')
