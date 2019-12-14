import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pickle
import sys
import numpy as np
import cv2

from sg2im import Sg2Image
from discriminators import ImageDiscriminator, ObjectDiscriminator
from loss import model_loss, gan_gen_loss, gan_dis_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_path = '/data/CoCo/processed_restricted/vocab.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

idx2obj = vocab.get('idx2obj')
idx2pred = vocab.get('idx2pred')
pred2idx = vocab.get('pred2idx')
num_objects = max(idx2obj.keys())
num_predicates = len(pred2idx)
print(f'Object counts: {len(idx2obj)}')
print(f'Predicate counts: {num_predicates}')

id_list = []; id_dict = {};
num_batches = 31
print('Loading idx...')
for i in range(num_batches):
    with open(f'/data/CoCo/val_processed_restricted/batch/{i+1}.pkl', 'rb') as f:
        batch = pickle.load(f)
        for iid in batch:
            id_list.append(iid)
id_list.sort()
for i in range(len(id_list)):
    id_dict[id_list[i]]=i
trans_list = []

def main():

    test_batches = load_images()

    model_path = sys.argv[1]
    model = torch.load(model_path)
    model.eval()

    img_discriminator = torch.load( 'img_discriminator.torch')
    obj_discriminator = torch.load( 'obj_discriminator.torch')


    for batch in test_batches:

        imgs_true = []
        object_indexs = []
        triples = []
        obj2img = []
        boxes_true = []
        masks_true = []

        image_num = 0
        offset = 0
        for image_id in batch:
            image = batch[image_id]
            imgs_true.append(image.get('image'))
            object_indexs.extend(image.get('objs'))
            triples.extend([[tri[0]+offset, tri[1], tri[2]+offset]
                            for tri in image.get('triples')])
            obj2img.extend([image_num for obj in image.get('objs')])
            boxes_true.extend(image.get('boxes'))
            masks_true.extend(image.get('masks'))
            image_num += 1
            offset += len(image.get('objs'))

        imgs_true = torch.Tensor(imgs_true).to(DEVICE)
        object_indexs = torch.LongTensor(object_indexs).to(DEVICE)
        triples = torch.LongTensor(triples).to(DEVICE)
        obj2img = torch.LongTensor(obj2img).to(DEVICE)
        boxes_true = torch.FloatTensor(boxes_true).to(DEVICE)
        masks_true = torch.FloatTensor(masks_true).to(DEVICE)

        img_discriminator.train()
        img_discriminator.zero_grad()
        obj_discriminator.train()
        obj_discriminator.zero_grad()

        ''' Forward pass. '''
        imgs_cheat_pred, boxes_pred, masks_pred = model(
            object_indexs, triples, obj2img, boxes_true, masks_true)
        imgs_pred,_,_ = model(object_indexs, triples, obj2img)
        imgs_scores = img_discriminator(imgs_pred)
        objs_scores, aux_loss = obj_discriminator(
            imgs_pred, object_indexs, boxes_true, obj2img)

        generator_loss = model_loss(
            imgs_true, imgs_pred, boxes_true, boxes_pred, masks_true, masks_pred)
        generator_img_loss = gan_gen_loss(imgs_scores)
        generator_obj_loss = gan_gen_loss(objs_scores)
        generator_aux_loss = aux_loss
        generator_total = generator_loss + generator_img_loss + \
            generator_obj_loss + generator_aux_loss

        print(f'Batch finishes...')
        save_batch_images(imgs_pred.cpu().detach().numpy(), 
                imgs_cheat_pred.cpu().detach().numpy(),
                batch)
        #torch.save(imgs_pred, f'out/image{epoch}.torch')
        
def save_batch_images(imgs_pred, imgs_cheat, batch):
    i = 0
    for iid in batch:
        img_true = batch[iid]['image']
        objs = batch[iid]['objs']
        triples = batch[iid]['triples']
        trans = [[idx2obj[objs[tri[0]]], idx2pred[tri[1]], [idx2obj[objs[tri[2]]]] ] for tri in triples]
        trans_list.append({'id':iid,'scene_graph':trans})
        img_pred = imgs_pred[i]
        img_cheat = imgs_cheat[i]
        i += 1
        name = 'img%06d' %id_dict[iid]
        save_image(img_true, f'image_true/{name}.png')
        save_image(img_pred, f'image_pred/{name}.png')
        save_image(img_cheat, f'image_cheat/{name}.png')
        with open(f'individual_sg/{name}.txt','w') as fsg:
            fsg.write(str(trans))
        
def save_image(raw_image, path):
    image = transform_image(raw_image)
    cv2.imwrite(path, image)

def transform_image(raw, size=64):
    image = np.zeros((size,size,3), dtype=np.uint8)
    for i in range(3):
        ch = raw[i]
        image[:,:,i] = np.round(255*(ch-np.min(ch))/(np.max(ch)-np.min(ch)))
    return image

def load_images():
    num_batches = 31
    train_batches = []
    print('Loading images...')
    for i in range(num_batches):
        batch_in = open(f'/data/CoCo/val_processed_restricted/batch/{i+1}.pkl', 'rb')
        batch = pickle.load(batch_in)
        train_batches.append(batch)
        batch_in.close()
    return train_batches


if __name__ == '__main__':
    main()
