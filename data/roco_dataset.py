import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption

class  roco_caption_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        filename = 'ann_train.json'
<<<<<<< HEAD
        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
=======
        ann_ori = json.load(open(os.path.join(ann_root,filename),'r'))
        anns = []
        for ann in ann_ori:
            image_path = os.path.join(image_root,ann['image'])
            try:   
                image = Image.open(image_path).convert('RGB')
                anns.append(ann)
            except:
                print(ann['image'])
        self.annotation = anns
>>>>>>> dc5a435b84dfa530f33d208a8e72cec809dc42ed
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
    def __len__(self):
        return len(self.annotation)
        # return 1024
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words)
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
         
        return image, caption, int(img_id)
    
    
class roco_caption_eval(Dataset):
<<<<<<< HEAD
    def __init__(self, transform, image_root, ann_root, split, max_words=30, prompt=''):  
        filenames = {'val':'ann_validation.json','test':'ann_test.json'}        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
=======
    def __init__(self, transform, image_root, ann_root, split):  
        filenames = {'val':'ann_validation.json','test':'ann_test.json'}        
        ann_ori = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        anns = []
        for ann in ann_ori:
            image_path = os.path.join(image_root,ann['image'])
            try:       
                image = Image.open(image_path).convert('RGB')
                anns.append(ann)
            except:
                print(ann['image'])
        self.annotation = anns
>>>>>>> dc5a435b84dfa530f33d208a8e72cec809dc42ed
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt
        self.labels = json.load(open('/home/wuyinjun/lzq/blip/MedFT/labels.json','r'))

    def __len__(self):
        # return len(self.annotation)
        return 1024
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words)
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        label = self.labels[index]['label']
        
        return label, image, caption,  int(img_id)
