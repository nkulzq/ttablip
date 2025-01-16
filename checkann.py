import json
from llava.eval.run_llava import load_image
import os

anns = json.load(open("/home/wuyinjun/lzq/roco-dataset/train_ann.json"))
anns_new = []
for ann in anns:
    image = os.path.join("/home/wuyinjun/lzq/roco-dataset", ann["image"])
    print(image)
    try:
        load_image(image)
        anns_new.append(ann)
    except:
        print("error")
        continue
json.dump(anns_new, open("/home/wuyinjun/lzq/roco-dataset/train_ann.json", 'w'))