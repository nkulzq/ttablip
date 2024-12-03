import json
import os

captions_root = '/home/wuyinjun/lzq/roco/roco-dataset'
folders = [
    'data/test/radiology',
    'data/test/non-radiology',
    'data/train/radiology',
    'data/train/non-radiology',
    'data/validation/radiology',
    'data/validation/non-radiology',
]
lines = []
for folder in folders:
    filename = os.path.join(captions_root, folder, 'captions.txt')
    with open(filename) as file:
        lines.extend([[line.rstrip('\n'), folder] for line in file])
captions = []
for line, folder in lines:
    line_parts_tab = line.split("\t")
    image = folder + '/' + line_parts_tab[0] + '.jpg'
    caption = line_parts_tab[1]
    captions.append({'image':image, 'caption':caption})
with open(os.path.join(captions_root, 'ann.json'), 'w') as f:  
    json.dump(captions, f) 