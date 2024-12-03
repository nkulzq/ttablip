import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

class  medicat_caption_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        filename = 'anns.json'
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.annotation = self.annotation[:int(0.8*len(self.annotation))]
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
    def __len__(self):
        return len(self.annotation)
        # return 1024
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['pdf_hash']+'_'+ann['fig_uri'])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['s2_caption'], self.max_words)
        img_id = ann['pdf_hash'] + '_' + ann['fig_uri']
         
        return image, caption, img_id
    
    
class medicat_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30, prompt=''):  
        filename = 'anns.json'
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        if split == 'test':
            self.annotation = self.annotation[int(0.9*len(self.annotation)):]
        else:
            self.annotation = self.annotation[int(0.8*len(self.annotation)): int(0.9*len(self.annotation))]
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['pdf_hash']+'_'+ann['fig_uri'])       
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        caption = self.prompt+pre_caption(ann['s2_caption'], self.max_words)
        img_id = ann['pdf_hash'] + '_' + ann['fig_uri']
        
        return image, caption, img_id
