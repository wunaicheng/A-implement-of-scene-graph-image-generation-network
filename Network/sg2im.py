import torch
import torch.nn as nn
import torch.nn.functional as F

from sg2layout import *
from layout2im import RefineModule, RefineNetwork


class Sg2Image(nn.Module):
    def __init__(self, num_objects, num_predicates):
        super(Sg2Image, self).__init__()
        self.obj_embeddings = nn.Embedding(num_objects, 128)
        self.pred_embeddings = nn.Embedding(num_predicates, 128)

        self.graphconv = GraphConvNetwork()
        self.box = BoxNetwork()
        self.mask = MaskNetwork()
        self.refine = RefineNetwork()

    def forward(self, object_indexs, triples, obj2img, boxes_true=None, masks_true=None):

        O = object_indexs.size(0)
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        edges = torch.stack([s, o], dim=1)

        things = self.obj_embeddings(object_indexs)
        predicates = self.pred_embeddings(p)

        things, predicates = self.graphconv(things, predicates, edges)
        boxes_pred = self.box(things)

        masks_pred = self.mask(things.view(O, 128, 1, 1))
        masks_pred = masks_pred.squeeze(1).sigmoid()

        boxes = boxes_pred if boxes_true is None else boxes_true
        masks = masks_pred if masks_true is None else masks_true

        layout = cast(things, boxes, masks, obj2img)
        imgs_pred = self.refine(layout)

        return imgs_pred, boxes_pred, masks_pred
