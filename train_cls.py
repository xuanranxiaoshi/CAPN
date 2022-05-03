import random
import numpy as np
import sys
import os
import argparse
import yaml
import datetime
from pathlib import Path
import logging
import importlib
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import provider
from data_util.ModelNet40Loader2 import ModelNet40 as ModelNet40_2
from data_util.ModelNet40Loader3 import ModelNet40 as ModelNet40_3
from data_util.ModelNet40Loader4 import ModelNet40 as ModelNet40_4



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

parser = argparse.ArgumentParser(description='Cls Training')
parser.add_argument('--config', default='cfgs/config_cls.yaml', type=str)

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def main():
    def log_string(str):
        logger.info(str)
        print(str)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('cls')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    # try:
    #     os.makedirs(args.save_path)
    # except OSError:
    #     pass

    '''LOG'''

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_name = '%s/%s_normal.txt' % (log_dir, args.model) if args.normal else '%s/%s.txt' % (log_dir, args.model)
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    dataset = args.data_root.split('/')[-1]
    if dataset=="shapenet":
        
        train_dataset = ShapeNetPart(root = args.data_root, num_points = args.npoint, split = 'trainval', normalize = True, normal = args.normal)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
        test_dataset = ShapeNetPart(root = args.data_root, num_points = args.npoint, split = 'test', normalize = True, normal = args.normal)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    elif dataset=="modelnet40_fps":
        
        train_dataset = ModelNet40_2(root = args.data_root,partition='train',normal = args.normal)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
        test_dataset = ModelNet40_2(root = args.data_root,partition='test',normal = args.normal)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    elif dataset=="modelnet40_fps_1200":
        
        train_dataset = ModelNet40_4(root = args.data_root,partition='train',normal = args.normal)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
        test_dataset = ModelNet40_2(root = args.data_root,partition='test',normal = args.normal)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    elif dataset=="modelnet40":
        
        train_dataset = ModelNet40_3(root = args.data_root,partition='train',normal = args.normal)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
        test_dataset = ModelNet40_3(root = args.data_root,partition='test',normal = args.normal)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)

      

    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of test data is: %d" %  len(test_dataset))

    num_classes = args.num_classes
    num_part = args.num_part

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))
    shutil.copy('models/%s_util.py' % args.model,str(experiment_dir))
    shutil.copy('models/curvenet_cls_util.py',str(experiment_dir))
    shutil.copy('./cfgs/config_cls.yaml', str(experiment_dir))
    shutil.copy('./train_cls.py', str(experiment_dir))

    #net = MODEL.get_model(num_part, normal_channel=args.normal)
    #classifier = nn.DataParallel(net).cuda()
    #classifier=nn.DataParallel(net,device_ids=[0,1])
    net = MODEL.get_model(num_classes, normal_channel=args.normal).cuda()
    classifier = nn.DataParallel(net).cuda()
    criterion = MODEL.get_loss().cuda()
    
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if(m.bias is not None):
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if(m.bias is not None):
                torch.nn.init.constant_(m.bias.data, 0.0)
    
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/'+args.checkpoint)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
        log_string('Load model successfully: %s' % (args.checkpoint))
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        log_string('optimizer: Adam')
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    elif args.optimizer == 'AdaBound':
        optimizer = adabound.AdaBound(classifier.parameters(), lr=args.learning_rate, final_lr=0.1)
    elif args.optimizer == 'SGD+COS':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate*100, momentum=0.9, weight_decay=args.decay_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-3)
    else:
        log_string('optimizer: SGD')
        #optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate*100, momentum=0.9, weight_decay=args.decay_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.cosepoch,T_mult=1)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
    


    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_loss = 1000

    global_epoch = 0


    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        mean_loss = []

        lr = optimizer.param_groups[0]['lr']
        log_string('Learning rate:%f' % lr)
        '''learning one epoch'''
        for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9):
            points, label = data
          
            points = points.data.numpy()
            if dataset=="modelnet40_fps_1200":
                idx = np.random.choice(1200,1024,replace=False)
                points = points[:,idx]
            if args.drop:
                points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3],scale_low=args.scale_low, scale_high=args.scale_high)
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3],shift_range=args.shift_range)
            points = torch.Tensor(points)
            points, label = points.float().cuda(),label.long().cuda()
            #label = to_categorical(label, num_classes)
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, label.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(label.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            mean_loss.append(loss.item() / float(points.size()[0]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1)
            optimizer.step()
        scheduler.step()
        train_instance_acc = np.mean(mean_correct)
        train_loss = np.mean(mean_loss)
        log_string('Train accuracy is: %.5f loss: %.12f' %(train_instance_acc,train_loss))


        with torch.no_grad():

            test_acc = []
            test_loss = []
            test_metrics = {}
            class_acc = np.zeros((num_classes, 3))
            for batch_id, (points, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), smoothing=0.9):
                points, label = points.float().cuda(), label.long().cuda()
                
                classifier = classifier.eval()
                pred, trans_feat = classifier(points)

                loss = criterion(pred, label.long(), trans_feat)
                pred_choice = pred.data.max(1)[1]

                for cat in np.unique(label.cpu()):
                    classacc = pred_choice[label == cat].eq(label[label == cat].long().data).cpu().sum()
                    class_acc[cat, 0] += classacc.item() / float(points[label == cat].size()[0])
                    class_acc[cat, 1] += 1

                correct = pred_choice.eq(label.long().data).cpu().sum()
                test_acc.append(correct.item() / float(points.size()[0]))
                test_loss.append(loss.item() / float(points.size()[0]))
            
            class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
            class_acc = np.mean(class_acc[:, 2])
            instance_acc = np.mean(test_acc)
            instance_loss = np.mean(test_loss)

            test_metrics['instance_acc'] = instance_acc 
            test_metrics['class_acc'] = class_acc
            test_metrics['loss'] = instance_loss

        log_string('Epoch %d Test Instance Accuracy: %f  Class Accuracy: %f  loss: %.12f' % (
                 epoch+1, test_metrics['instance_acc'],test_metrics['class_acc'],test_metrics['loss']))
        if (test_metrics['instance_acc'] >= best_instance_acc):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_normal_model.pth' if args.normal else str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_instance_acc': test_metrics['instance_acc'],
                'test_class_acc': test_metrics['class_acc'],
                'test_loss': test_metrics['loss'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if (test_metrics['instance_acc'] >= best_instance_acc):
            best_instance_acc = test_metrics['instance_acc']
        if test_metrics['class_acc'] > best_class_acc:
            best_class_acc = test_metrics['class_acc']
        if test_metrics['loss'] <= best_loss:
            best_loss = test_metrics['loss']
        log_string('Best instance_acc is: %.5f'%best_instance_acc)
        log_string('Best class_acc is: %.5f'%best_class_acc)
        log_string('Best loss is: %.12f'%best_loss)
        global_epoch+=1


       
if __name__ == "__main__":
    main()