import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class Embedding(nn.Module):
    def __init__(self, backbone="resnet50", embedding_dim=512, pretrained=True):
        super().__init__()
        #just using timm because its easy
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        feats = self.backbone(x)
        emb = self.embedding(feats)
        #l2 normalize so cosine sim is just a dot product later
        emb = F.normalize(emb, p=2, dim=1)
        #print(emb.shape)
        return emb