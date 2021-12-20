
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 9 14:13:59 2021
classnames_v1_5 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 
                   'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank', 
                   'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 
                   'container-crane']

classnames_v1_0 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 
                   'large-vehicle', 'ship', 'tennis-court','basketball-court', 'storage-tank', 
                   'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

@author: selene
"""


import argparse
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True

import test 
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *

mixed_precision = False

hyp = {'optimizer': 'SGD',  
       'lr0': 0.01,  
       'momentum': 0.937,  
       'weight_decay': 5e-4,  
       'giou': 0.05, 
       'cls': 0.5, 
       'cls_pw': 1.0, 
       'obj': 1.0,  
       'obj_pw': 1.0,  
       'iou_t': 0.20,  
       'anchor_t': 4.0, 
       'fl_gamma': 0.0, 
       'hsv_h': 0.015,  
       'hsv_s': 0.7,  
       'hsv_v': 0.4,  
       'degrees': 0.0,  
       'translate': 0.0,  
       'scale': 0.5,  
       'shear': 0.0}


def train(hyp, tb_writer, opt, device):
    log_dir = tb_writer.log_dir if tb_writer else 'runs/evolution'  
    wdir = str(Path(log_dir) / 'weights') + os.sep  
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = log_dir + os.sep + 'results.txt'
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.local_rank
    with open(Path(log_dir) / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(Path(log_dir) / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])
    if rank in [-1, 0]:
        for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
            os.remove(f)

    model = Model(opt.cfg, nc=nc).to(device)

    gs = int(max(model.stride)) 
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size] 
    nbs = 64  
    accumulate = max(round(nbs / total_batch_size), 1)  
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v) 
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v) 
            else:
                pg0.append(v)  

    if hyp['optimizer'] == 'adam':  
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999)) 
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']}) 
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2
    
   
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2 
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    with torch_distributed_zero_first(rank):
        google_utils.attempt_download(weights)
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt'): 
        ckpt = torch.load(weights, map_location=device)  

        # load model
        try:
            exclude = ['anchor'] 
            ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if k in model.state_dict() and not any(x in k for x in exclude)
                             and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(ckpt['model'], strict=False)
            print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(model.state_dict()), weights))
        except KeyError as e:
            s = "%s is not compatible with %s. model differences or %s may be out of date. " 
                % (weights, opt.cfg, weights, weights)
            raise KeyError(s) from e

        
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results']) 

        
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
#            print('%s was trained for %g epochs. Fine-tuning for %g additional epochs.' %
#                  (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch'] 

        del ckpt
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    if device.type != 'cpu' and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    
    if opt.sync_bn and device.type != 'cpu' and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print('Using SyncBatchNorm()')

    
    ema = torch_utils.ModelEMA(model) if rank in [-1, 0] else None

   
    if device.type != 'cpu' and rank != -1:
        model = DDP(model, device_ids=[rank], output_device=rank)

   
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
                                            cache=opt.cache_images, rect=opt.rect, local_rank=rank,
                                            world_size=opt.world_size)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader) 
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt, hyp=hyp, augment=False,
                                       cache=opt.cache_images, rect=True, local_rank=-1, world_size=opt.world_size)[0]
    hyp['cls'] *= nc / 80. 
    model.nc = nc  
    model.hyp = hyp 
    model.gr = 1.0 
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    if rank in [-1, 0]:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0]) 
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        plot_labels(labels, save_dir=log_dir)
        if tb_writer:
            # tb_writer.add_hparams(hyp, {}) 
            tb_writer.add_histogram('classes', c, 0)

        
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    
    t0 = time.time()
    nw = max(3 * nb, 1e3) 
    maps = np.zeros(nc)  
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  
    if rank in [0, -1]:
        print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
        print('Using %g dataloader workers' % dataloader.num_workers)
        print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  
        torch.cuda.empty_cache()
        model.train()

        if dataset.image_weights:
            
            if rank in [-1, 0]:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2 
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                                 k=dataset.n) 
            
            if rank != -1:
                indices = torch.zeros([dataset.n], dtype=torch.int)
                if rank == 0:
                    indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()


        mloss = torch.zeros(4, device=device) 
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  
            ni = i + nb * epoch  
            imgs = imgs.to(device, non_blocking=True).float() / 255.0 

            if ni <= nw:
                xi = [0, nw] 
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs 
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]] 
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
            if rank != -1:
                loss *= opt.world_size 
            if not torch.isfinite(loss):
                print('non-finite loss', loss_items)
                return results

            # Backward
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f = str(Path(log_dir) / ('train_batch%g.jpg' % ni))  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            
        scheduler.step()

        if rank in [-1, 0]:
            
            if ema is not None:
                ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:
                results, maps, times = test.test(opt.data,
                                                 batch_size=total_batch_size,
                                                 imgsz=imgsz_test,
                                                 save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                                 model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=log_dir)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Tensorboard
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)

           
            fi = fitness(np.array(results).reshape(1, -1))  
            if fi > best_fitness:
                best_fitness = fi

            
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module if hasattr(ema, 'module') else ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                
                torch.save(ckpt, last)
                if best_fitness == fi: 
                    torch.save(ckpt, best)
                del ckpt


    if rank in [-1, 0]:
        
        n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None 
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload
        # Finish
        if not opt.evolve:
            plot_results(save_dir=log_dir)  
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='dota_data/yolov5l_dota.yaml', help='model.yaml path')  # default='models/yolov5s.yaml'  模型yaml文件地址
    parser.add_argument('--data', type=str, default='dota_data/dota_name.yaml', help='data.yaml path')  # default='data/coco128.yaml'       数据yaml文件地址
    parser.add_argument('--hyp', type=str, default='', help='hyp.yaml path (optional)')
    parser.add_argument('--epochs', type=int, default=250)  # default=300
    parser.add_argument('--batch-size', type=int, default=8, help="Total batch size for all gpus.")  # default=16
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False, help='resume from given path/to/last.pt, or most recent run if blank.')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov5l.pt', help='initial weights path')  # weights
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # gpu
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    opt = parser.parse_args()

    last = get_latest_run() if opt.resume == 'get_last' else opt.resume 
    if last and not opt.weights:
        print(f'Resuming training from {last}')
    opt.weights = last if opt.resume and not opt.weights else opt.weights
    if opt.local_rank in [-1, 0]:
        check_git_status()
    opt.cfg = check_file(opt.cfg)  
    opt.data = check_file(opt.data) 
    if opt.hyp: 
        opt.hyp = check_file(opt.hyp)  
        with open(opt.hyp) as f:
            hyp.update(yaml.load(f, Loader=yaml.FullLoader)) 
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size))) 
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    opt.total_batch_size = opt.batch_size
    opt.world_size = 1
    if device.type == 'cpu':
        mixed_precision = False
    elif opt.local_rank != -1:
        # DDP mode
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://') 

        opt.world_size = dist.get_world_size()
        assert opt.batch_size % opt.world_size == 0, "Batch size is not a multiple of the number of devices given!"
        opt.batch_size = opt.total_batch_size // opt.world_size
    print(opt)

    # Train
    if not opt.evolve:
        if opt.local_rank in [-1, 0]:
            print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
            tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', opt.name))
        else:
            tb_writer = None
        train(hyp, tb_writer, opt, device)

    #(optional)
    else:
        assert opt.local_rank == -1, "DDP mode currently not implemented for Evolve!"

        tb_writer = None
        opt.notest, opt.nosave = True, True  
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  

        for _ in range(10):
            torch.cuda.empty_cache()
            if os.path.exists('evolve.txt'):  
                # Select parent(s)
                parent = 'single' 
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    
                    x = x[random.choices(range(n), weights=w)[0]] 
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  

                # Mutate
                mp, s = 0.9, 0.2  
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                v = np.ones(ng)
                while all(v == 1):  
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()): 
                    hyp[k] = x[i + 7] * v[i]  

            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            torch.cuda.empty_cache()
            results = train(hyp.copy(), tb_writer, opt, device)
            print_mutation(hyp, results, opt.bucket)

