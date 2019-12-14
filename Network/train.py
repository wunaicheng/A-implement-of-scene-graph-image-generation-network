import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pickle

from sg2im import Sg2Image
from discriminators import ImageDiscriminator, ObjectDiscriminator
from loss import model_loss, gan_gen_loss, gan_dis_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print('Reading vocab...')
    vocab_in = open('/data/CoCo/processed_restricted/vocab.pkl', 'rb')
    vocab = pickle.load(vocab_in)
    vocab_in.close()

    idx2obj = vocab.get('idx2obj')
    pred2idx = vocab.get('pred2idx')
    num_objects = max(idx2obj.keys())
    num_predicates = len(pred2idx)
    print(f'Object counts: {len(idx2obj)}')
    print(f'Predicate counts: {num_predicates}')

    print('Building Sg2Image model...')
    model = Sg2Image(1+num_objects, 1+num_predicates).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    img_discriminator = ImageDiscriminator().to(DEVICE)
    img_optimizer = torch.optim.Adam(img_discriminator.parameters(), lr=0.0002)
    obj_discriminator = ObjectDiscriminator(num_objects).to(DEVICE)
    obj_optimizer = torch.optim.Adam(obj_discriminator.parameters(), lr=0.0002)

    # print(model)
    # print(img_discriminator)
    # print(obj_discriminator)

    train_batches = load_images()

    epoches = 50
    for epoch in range(epoches):
        print(f'Epoch {epoch} begins...')
        batch_num = 0
        #loss = 0
        for batch in train_batches:

            if batch_num % 100 == 0:
                print(f'Training batch {batch_num}...')

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

            model.train()
            model.zero_grad()
            img_discriminator.train()
            img_discriminator.zero_grad()
            obj_discriminator.train()
            obj_discriminator.zero_grad()

            ''' Forward pass. '''
            imgs_pred, boxes_pred, masks_pred = model(
                object_indexs, triples, obj2img, boxes_true, masks_true)
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
            #loss += generator_total

            if not math.isfinite(generator_total):
                print('WARNING: Got infinite generator loss. Not backpropagating.')
                continue

            ''' Generator back-propagation. '''
            optimizer.zero_grad()
            generator_total.backward(retain_graph=True)
            optimizer.step()

            ''' Discriminator back-propagation. '''
            scores_fake = img_discriminator(imgs_pred)
            scores_real = img_discriminator(imgs_true)
            discriminator_img_loss = gan_dis_loss(scores_real, scores_fake)
            img_optimizer.zero_grad()
            discriminator_img_loss.backward(retain_graph=True)
            img_optimizer.step()

            scores_fake, aux_fake = obj_discriminator(
                imgs_pred, object_indexs, boxes_pred, obj2img)
            scores_real, aux_real = obj_discriminator(
                imgs_true, object_indexs, boxes_true, obj2img)
            discriminator_obj_loss = gan_dis_loss(
                scores_real, scores_fake) + aux_fake + aux_real
            obj_optimizer.zero_grad()
            discriminator_obj_loss.backward(retain_graph=True)
            obj_optimizer.step()
            
            batch_num += 1

        print(f'Epoch {epoch} finishes...')
        torch.save(imgs_pred, f'out/image{epoch}.torch')
        #if epoch == 50:
        #    torch.save(model, '/data/CoCo/processed_restricted/out/half-model.torch')
        

    print('Saving model...')
    torch.save(model, 'model.torch')
    torch.save(img_discriminator, 'img_discriminator.torch')
    torch.save(obj_discriminator, 'obj_discriminator.torch')
    return model


def load_images():
    num_batches = 773
    train_batches = []
    print('Loading images...')
    for i in range(num_batches):
        batch_in = open(f'/data/CoCo/processed_restricted/batch/{i+1}.pkl', 'rb')
        batch = pickle.load(batch_in)
        train_batches.append(batch)
        batch_in.close()
    return train_batches


if __name__ == '__main__':
    model = main()
