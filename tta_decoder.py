'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
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


def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 2

    results = []
    for i, (image, caption, image_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        model.eval()
        image = image.to(device)
        image_embeds = model.model.get_image_embed(image).cpu().detach()
        caption = model.model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], min_length=config['min_length'])
        model.train()
        caption_embeds = model.model.get_caption_embed(caption, device)
        loss = model(image_embeds, caption_embeds, 'image')
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=model.optimizer.param_groups[0]["lr"])
        
        # for caption, img_id in zip(captions, image_id):
        #     if '[unused0]' in caption:
        #         raise ValueError("Error: The token '[unused0]' was found in the caption, interrupting the process.")
        #     results.append({"image_id": img_id.item(), "caption": caption})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, config):
   # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for _, image, caption, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)
        image_embeds = model.model.get_image_embed(image)
        captions = model.model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], min_length=config['min_length'])
        
        for caption, img_id, image_embed in zip(captions, image_id, image_embeds):
            result.append({"image_id": img_id.item(), "caption": caption})
    return result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_roco', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [False,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[0,0,0],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])         

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])
    model.requires_grad_(False)
    model.text_decoder.requires_grad_(True)
    total_params = sum(p.numel() for p in model.parameters())  
    print(f"Total number of parameters: {total_params}")   
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"Total number of trainable parameters: {trainable_params}")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    # cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
    optmodel = OptModel(model, optimizer, config, device)
    optmodel = optmodel.to(device)   
    
    model_without_ddp = optmodel
    # if args.distributed:
    #     optmodel = torch.nn.parallel.DistributedDataParallel(optmodel, device_ids=[args.gpu])
    #     model_without_ddp = optmodel.module    
            
    # best = 0
    # best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(100):
        
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
        train_stats = train(model_without_ddp, train_loader, optimizer, epoch, device, config)
        # json.dump(val_results, open("./opt_results.json", 'w'))
        
        val_result = evaluate(model_without_ddp, val_loader, device, config)
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')
        # test_result = evaluate(model_without_ddp, test_loader, device, config)  
        # test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='image_id')  

        if utils.is_main_process():   
            # coco_val = coco_caption_eval(config['ann_root'],val_result_file,'val')
            # coco_test = coco_caption_eval(config['ann_root'],test_result_file,'test')
            
            if args.evaluate:            
                log_stats = {**{f'val_{k}': v for k, v in coco_val.items()},
                             **{f'test_{k}': v for k, v in coco_test.items()},                       
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                   
            else:             
                save_obj = {
                    'model': model_without_ddp.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_{}.pth'.format(epoch)))
                # json.dump(model_without_ddp.losses, open("./losses.json", 'w'))
                # if coco_val['CIDEr'] + coco_val['Bleu'][3] > best:
                #     best = coco_val['CIDEr'] + coco_val['Bleu'][3]
                #     best_epoch = epoch                
                    # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                #              **{f'val_{k}': v for k, v in coco_val.items()},
                #              **{f'test_{k}': v for k, v in coco_test.items()},                       
                             'epoch': epoch,
                #              'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
                    
        if args.evaluate:
            break
        dist.barrier()

    # print(optmodel.train_batch)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_roco.yaml')
    parser.add_argument('--output_dir', default='output/tta_decoder')        
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
