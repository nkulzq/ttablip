import torch
import torch.nn as nn
from torchclustermetrics import silhouette
from transformers import GPT2Tokenizer, GPT2Model
from transformers import BertTokenizer, BertModel
import numpy as np
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from collections import Counter
from transformers import AutoTokenizer, AutoModel

class OptModel(nn.Module):
    def __init__(self, model, optimizer, config, device, steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.steps = steps
        # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # self.inverse_model = GPT2Model.from_pretrained('gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.inverse_model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        self.inverse_model.eval()
        self.losses = []
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"

    
    def forward_and_adapt(self, image_embeds, captions, mode):
        """Forward and adapt model on batch of data.

        Measure entropy of the model prediction, take gradients, and update params.
        """
        # image_embeds = model.get_image_embed(images)
        # adapt
        if mode == 'image':
            caption_embeds = captions
            clustering = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(image_embeds)
            labels = clustering.labels_
            min_samples = 3
            label_counts = Counter(labels)
            satisfies_min_samples = all(count >= min_samples for count in label_counts.values())
            loss = 1 - silhouette.score(caption_embeds, labels)
            if satisfies_min_samples:
                print('step')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                print('error')
                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
        else:
            raise ValueError("training mode error: not using image labels")
        
        return loss
    
    def forward(self, image_embeds, captions, mode):
        for _ in range(self.steps):
            loss = self.forward_and_adapt(image_embeds, captions, mode)
            self.losses.append(loss)
        return loss


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(True)
    # configure norm for tent updates: enable grad + force batch statisics
    # model.visual_encoder.requires_grad_(True)
    # for m in model.modules():
        # if isinstance(m, nn.VisionTransformer):
        #     m.requires_grad_(True)
        #     force use of batch stats in train and eval modes
        #     m.track_running_stats = False
        #     m.running_mean = None
        #     m.running_var = None
    return model