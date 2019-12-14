#!/usr/bin/python3

import os, math, json, random
import numpy as np
import torch
import pickle
from matplotlib.image import imread
from skimage.transform import resize
import pycocotools.mask as mask_utils
from collections import defaultdict

#Notes about adjustments from the original version
#stuff_json required
#mask_size neglected, same as the image size
#

class CocoDataset():
    def __init__(self, image_path, instance_path, stuff_path, image_size=64, mask_size=16,
            min_obj_size=0.02, min_obj_occurance=4000, min_nobj=3, max_nobj=8):

        self.image_path = image_path
        self.image_size = image_size
        self.mask_size = mask_size
        self.all_image_data = {}
        
        with open(instance_path, 'r') as f:
            instance_data = json.load(f)
        with open(stuff_path, 'r') as f:
            stuff_data = json.load(f)

        self.image_ids = []
        self.image_id2f = {}
        self.image_id2size = {}
        for image_data in instance_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id2f[image_id] = filename
            self.image_id2size[image_id] = (width, height)

        #construct vocab
        self.obj2idx = {}
        self.idx2obj = {}
        self.pred2idx = {'left of':1, 'right of':2, 'above':3, 'below':4, 'inside':5,
                'surrounding':6}
        self.idx2pred = [None, 'left of', 'right of', 'above', 'below', 'inside', 'surrounding']

        #count object occurance
        count_obj = {}
        for obj_data in instance_data['annotations']:
            if(obj_data['category_id'] in count_obj):
                count_obj[obj_data['category_id']] += 1
            else:
                count_obj[obj_data['category_id']] = 1
        for obj_data in stuff_data['annotations']:
            if(obj_data['category_id'] in count_obj):
                count_obj[obj_data['category_id']] += 1
            else:
                count_obj[obj_data['category_id']] = 1

        for categ_data in instance_data['categories']:
            categ_id = categ_data['id']
            categ_name = categ_data['name']
            if(count_obj[categ_id]>min_obj_occurance):
                self.obj2idx[categ_name] = categ_id
                self.idx2obj[categ_id] = categ_name
        for categ_data in stuff_data['categories']:
            categ_id = categ_data['id']
            categ_name = categ_data['name']
            if(count_obj[categ_id]>min_obj_occurance):
                self.obj2idx[categ_name] = categ_id
                self.idx2obj[categ_id] = categ_name

        #bounding box data
        self.image_id2obj = defaultdict(list)
        for obj_data in instance_data['annotations']:
            image_id = obj_data['image_id']
            _, _, w, h = obj_data['bbox']
            W, H = self.image_id2size[image_id]
            box_area = (w * h) / (W * H)
            box_check = box_area > min_obj_size
            categ_check = obj_data['category_id'] in self.idx2obj
            #categ_check = self.idx2obj[obj_data['category_id']] in self.obj2idx
            #if(box_check and categ_check):
            if(box_check):
                self.image_id2obj[image_id].append(obj_data)
        for obj_data in stuff_data['annotations']:
            image_id = obj_data['image_id']
            _, _, w, h = obj_data['bbox']
            W, H = self.image_id2size[image_id]
            box_area = (w * h) / (W * H)
            box_check = box_area > min_obj_size
            categ_check = obj_data['category_id'] in self.idx2obj
            #categ_check = self.idx2obj[obj_data['category_id']] in self.obj2idx
            #if(box_check and categ_check):
            if(box_check):
                self.image_id2obj[image_id].append(obj_data)
            #this part may require further examination

        #check object number
        new_image_ids = []
        print('#image before object check:', len(self.image_ids))
        count = 0
        for image_id in self.image_ids:
            num_objs = 0; valid = True;
            for obj in self.image_id2obj[image_id]:
                if(not obj['category_id'] in self.idx2obj):
                    valid = False
                    break
                num_objs += 1 
            #num_objs = len(self.image_id2obj[image_id])
            if(min_nobj<= num_objs <=max_nobj and valid):
                new_image_ids.append(image_id)
            if(not valid):
                count += 1
        print('# eliminated in object:', count)
        print('# left:',len(new_image_ids))
        self.image_ids = new_image_ids

        '''
        #check image dimension
        new_image_ids = []
        count = 0
        for image_id in self.image_ids:
            filename = self.image_id2f[image_id]
            path = os.path.join(self.image_path, filename)
            image = imread(path)
            if(len(image.shape)==3 and image.shape[2]==3):
                new_image_ids.append(image_id)
                count += 1
                print(count)
        self.image_ids = new_image_ids
        '''

    def get_data(self, image_id):
        #read image
        filename = self.image_id2f[image_id]
        path = os.path.join(self.image_path, filename)
        image = imread(path)
        HH, WW, _ = image.shape
        image = resize(image, (self.image_size, self.image_size), mode='constant')
        #stack dimension differently
        print('image:',image.min(), image.max())
        new_image = np.zeros((3, self.image_size, self.image_size))
        for i in range(3):
            new_image[i] = image[:,:,i]
        image = new_image

        #read bbox and mask
        H,W = self.image_size, self.image_size
        objs, boxes, masks = [],[],[]
        for obj_data in self.image_id2obj[image_id]:
            objs.append(obj_data['category_id'])
            x, y, w, h = obj_data['bbox']
            x0 = x / WW
            y0 = y / HH
            x1 = (x + w) / WW
            y1 = (y + h) / HH
            boxes.append(np.array([x0,y0,x1,y1]))

            mask = seg_to_mask(obj_data['segmentation'], WW, HH)
            mx0, mx1 = int(round(x)), int(round(x + w))
            my0, my1 = int(round(y)), int(round(y + h))
            mx1 = max(mx0 + 1, mx1)
            my1 = max(my0 + 1, my1)
            mask = mask[my0:my1, mx0:mx1]
            mask = resize(255.0 * mask, (self.mask_size, self.mask_size), mode='constant')
            masks.append((mask > 128).astype(np.int64))

        #calculate object center
        obj_centers = []
        MH, MW = masks[0].shape
        for i, obj_idx in enumerate(objs):
            x0, y0, x1, y1 = boxes[i]
            mask = (masks[i] == 1)
            xs,ys = np.meshgrid(np.arange(MW), np.arange(MH))
            if(mask.sum() == 0):
                mean_x = 0.5 * (x0 + x1)
                mean_y = 0.5 * (y0 + y1)
            else:
                mean_x = xs[mask].mean()
                mean_y = ys[mask].mean()
            obj_centers.append([mean_x,mean_y])
        obj_centers = np.array(obj_centers)

        #generate scene graph
        triples = []
        num_objs = len(objs)
        for i, cur in enumerate(objs):
            choices = [j for j,obj in enumerate(objs) if j != i and obj!=cur]
            j = random.choice(choices)
            if(random.random() > 0.5):
                s, o = i,j
            else:
                s, o = j,i

            sx0, sy0, sx1, sy1 = boxes[s]
            ox0, oy0, ox1, oy1 = boxes[o]
            d = obj_centers[s] - obj_centers[o]
            theta = math.atan2(d[1], d[0])

            if(sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1):
                p = 'surrounding'
            elif(sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1):
                p = 'inside'
            elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                p = 'left of'
            elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                p = 'above'
            elif -math.pi / 4 <= theta < math.pi / 4:
                p = 'right of'
            elif math.pi / 4 <= theta < 3 * math.pi / 4:
                p = 'below'
            p = self.pred2idx[p]
            triples.append([s, p, o])
        #this part is for establishing connection between all objects
        '''
        for i in range(num_objs):
            for j in range(i+1,num_objs):
                if(random.random() > 0.5):
                    s, o = i,j
                else:
                    s, o = j,i

                sx0, sy0, sx1, sy1 = boxes[s]
                ox0, oy0, ox1, oy1 = boxes[o]
                d = obj_centers[s] - obj_centers[o]
                theta = math.atan2(d[1], d[0])

                if(sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1):
                    p = 'surrounding'
                elif(sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1):
                    p = 'inside'
                elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                    p = 'left of'
                elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                    p = 'above'
                elif -math.pi / 4 <= theta < math.pi / 4:
                    p = 'right of'
                elif math.pi / 4 <= theta < 3 * math.pi / 4:
                    p = 'below'
                p = self.pred2idx[p]
                #triples.append([objs[s], p, objs[o]])
                #s and o are indexs of the objects in that scene graph
                triples.append([s, p, o])
        '''
        return image, objs, boxes, masks, triples

    def extract_all_images(self):
        for image_id in self.image_ids:
            image, objs, boxes, masks, triples = self.get_data(image_id)
            self.all_image_data[image_id] = {'image':image, 'objs':objs, 'boxes':boxes, 
                    'masks':masks, 'triples':triples}


def seg_to_mask(seg, width=1.0, height=1.0):
    if(type(seg)==list):
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif(type(seg['counts']) == list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle)

def save_all_data(coco_dir, dataset, batch_size=32):
    vocab = {'obj2idx':dataset.obj2idx, 'pred2idx':dataset.pred2idx, 
            'idx2obj':dataset.idx2obj, 'idx2pred':dataset.idx2pred}
    vocab_path = os.path.join(coco_dir, 'processed_restricted/vocab.pkl')
    with open(vocab_path, 'wb') as out:
        pickle.dump(vocab, out)
    print('vocab saved!')
    count = 0
    batch = {}
    for image_id in dataset.image_ids:
        try:
            image, objs, boxes, masks, triples = dataset.get_data(image_id)
        except:
            continue
        batch[image_id] = {'image':image, 'objs':objs, 'boxes':boxes,
                    'masks':masks, 'triples':triples}
        count += 1
        #for continuation:
        #if(count // batch_size <198 ):
        #    continue
        if(count % batch_size == 0):
            batch_path = os.path.join(coco_dir, f'processed_restricted/batch/{count//batch_size}.pkl')
            with open(batch_path, 'wb') as out:
                pickle.dump(batch, out)
            print(f'batch #{count//batch_size} saved!')
            batch = {};


if(__name__=='__main__'):
    COCO_DIR = os.path.expanduser('/data/CoCo')
    image_path = os.path.join(COCO_DIR, 'images/train2017')
    instance_path = os.path.join(COCO_DIR, 'annotations/instances_train2017.json')
    stuff_path = os.path.join(COCO_DIR, 'annotations/stuff_train2017.json')

    #construct coco dataset class
    train_data = CocoDataset(image_path, instance_path, stuff_path)
    #vocab:     train_data.obj2idx, train_data.idx2obj
    #vocab:     train_data.pred2idx, train_data.idx2pred
    #image ids: train_data.image_ids
    
    print('len image_id:', len(train_data.image_ids))
    print('len obj:', len(train_data.idx2obj))

    #generate data for each image
    sample_id = train_data.image_ids[0]
    image, objs, boxes, masks, triples = train_data.get_data(sample_id)
    #image: np array (size,size,3)
    #objs: list of object idx
    #boxes: list of four corners of the bounding box (x0,y0,x1,y1) in fraction 0<x,y<1
    #masks: list of np arrays (size,size)
    #triples: list of [s,p,o] triples, s,o are object idx, p predicate idx
    
    print('image shape',image.shape)
    print('objs len', len(objs))
    print('objects:', objs)
    print('boxes len', len(boxes))
    print('boxes:',boxes)
    print('masks', len(masks))
    print('mask shape', masks[0].shape)
    print('triples:')
    print(triples,'\n')
    #for triple in triples:
    #    print(train_data.idx2obj[triple[0]], train_data.idx2pred[triple[1]], train_data.idx2obj[triple[2]])

    #save_all_data(COCO_DIR, train_data, 50)
    '''
    print('Saving all image data...')
    train_data.extract_all_images()
    print('length of image data:', len(train_data.all_image_data))
    pkl_path = os.path.join(COCO_DIR, 'train_data.pkl')
    with open(pkl_path, 'rb') as out:
        pickle.dump(train_data, out)
    print(f'Object saved at {pkl_path}')
    '''

