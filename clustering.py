import os
import json
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
from data.utils import pre_caption
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from data.roco_dataset import roco_caption_train, roco_caption_eval
from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
from data.nocaps_dataset import nocaps_eval
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.vqa_dataset import vqa_dataset
from data.nlvr_dataset import nlvr_dataset
from data.pretrain_dataset import pretrain_dataset
from transform.randaugment import RandomAugment
from sklearn.cluster import KMeans
import argparse
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval
from opt.opt import OptModel
from opt.opt import configure_model


def main(args, config):
    device = torch.device(args.device)
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([                        
                transforms.RandomResizedCrop(config['image_size'],scale=(0.5, 1.0),interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    annotation = json.load(open(os.path.join('/home/wuyinjun/lzq/roco/ann_validation.json'),'r'))
    group_size = 32
    annos = [annotation[i:i+group_size] for i in range(0, len(annotation), group_size)]
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                               vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                               prompt=config['prompt'])
    model.to(device)
    model.eval()
    results = []
    for i, anns in enumerate(annos):
        print(i)
        images = []
        img_ids = []
        for ann in anns:
            img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
            image_path = os.path.join('/home/wuyinjun/lzq/roco',ann['image'])        
            img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
            image = Image.open(image_path).convert('RGB')   
            image = transform_train(image)
            image = image.to(device)
            images.append(image)
            img_ids.append(img_id)
        images = torch.stack(images, dim=0)
        image_embeds = model.get_image_embed(images).cpu().detach()
        clustering = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(image_embeds)
        labels = clustering.labels_
        for img_id, label in zip(img_ids, labels):
            results.append({"image_id": int(img_id), "label": int(label)})
    json.dump(results, open("./labels.json", 'a'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_roco.yaml')
    parser.add_argument('--output_dir', default='output/test')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)