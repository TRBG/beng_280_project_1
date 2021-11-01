# General Imports
import argparse
import os
from collections import OrderedDict
from glob import glob
from tqdm import tqdm
import yaml
from datetime import datetime

# Scientific Computing Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib. pyplot as plt

# Pytorch Imports
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# Local Scripts Imports
import models
from dataset import Dataset
from metrics import mse_score
from utilities import LogMeter, count_params, plot_scans_and_reconstructions

models_names = models.__all__


def parse_args():
    """Fucntion to parse the arguments (settings) needed for the amin trianing script)"""
  
    # General settings
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', default=None,
                        help='The Default experiment (run) name: (default: model and date)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=16, type=int,
                        metavar='N', help='batch size (default: 16)')
    
    # Dataset Settings
    parser.add_argument('--dataset', default='ct',
                        help='dataset name')
    parser.add_argument('--scan_ext', default='.png',
                        help='scan file extension')
    parser.add_argument('--singl_v_bp_ext', default='.png',
                        help='single view back projections file extension')
    
    # Model Settings
    parser.add_argument('--model', '-a', metavar='MODEL', default='CT_Recon_Net',
                        choices = models_names)
    parser.add_argument('--input_channels', default=16, type=int,
                        help='input channels (represent the different number of views)')
    parser.add_argument('--input_w', default=64, type=int,
                        help='scan width')
    parser.add_argument('--input_h', default=64, type=int,
                        help='scan height')

    # Optimizer Setting
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # Scheduler Settings
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    
    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': LogMeter(),
                  'MSE': LogMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        
        input = input.cuda()
        target = target.cuda()

        # Generate the recosntruction
        output = model(input)
        loss = criterion(output, target)
        MSE = mse_score(output, target)

        # Computing the gradient and updating the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['MSE'].update(MSE, input.size(0))

        live_metrics = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('MSE', avg_meters['MSE'].avg),
        ])
        pbar.set_postfix(live_metrics)
        pbar.update(1)

        if np.random.uniform() >0.99:
                plot_scans_and_reconstructions(output, target)

    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('MSE', avg_meters['MSE'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': LogMeter(),
                  'MSE': LogMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            
            output = model(input)
            loss = criterion(output, target)
            MSE = mse_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['MSE'].update(MSE, input.size(0))

            live_metrics = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('MSE', avg_meters['MSE'].avg),
            ])
            pbar.set_postfix(live_metrics)
            pbar.update(1)

            if np.random.uniform() >0.95:
                plot_scans_and_reconstructions(output, target)

        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('MSE', avg_meters['MSE'].avg)])                        


def main():
  	
    # Getting the settings for the script
    config = vars(parse_args())
    
    config['dataset'] = config['dataset'] + '_' + str(config['input_channels']) + '_views'

    # Creating the subfolder where the results of the training will be stored
    if config['experiment_name'] is None:
      date_and_time = datetime.utcnow().strftime("date_%Y_%m_%d_time_%H_%M_%S")
      config['experiment_name'] = '%s_%s_%s' % (config['dataset'], config['model'], date_and_time)
    
    os.makedirs('experiments/%s' % config['experiment_name'], exist_ok=True)

    print('=' * 30)
    print('The options (settings) used for this script are:')
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('=' * 30)

    # Sacing the options to a YML file
    with open('experiments/%s/config.yml' % config['experiment_name'], 'w') as f:
        yaml.dump(config, f)
          
          
    # >>>>>>>>>>>>----------------------------->> Stopped Here
    # >>>>>>>>>>>>----------------------------->> Stopped Here
    # Defining the loss function used (criterion)
    criterion = nn.MSELoss().cuda()

    cudnn.benchmark = True

    # Creating and defining model, optimizer, and scheduler
    print("-- creating the model: %s" % config['model'])
    model = models.__dict__[config['model']](config['input_channels'])
    model = model.cuda()
          
    num_of_parms = count_params(model)
    print('=' * 30)
    print("The number of parameters in the model are: {:,}".format(num_of_parms))
    print('=' * 30)


    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              weight_decay=config['weight_decay'])


    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None

    # Setting up the dataset and data loading
    scans_list = glob(os.path.join('/content/data_dir', config['dataset'], 'scans', '*' + config['scan_ext']))
    scans_list = [os.path.splitext(os.path.basename(p))[0] for p in scans_list]

    train_scans_list, val_scans_list= train_test_split(scans_list, test_size=0.2, random_state=41)

    print('=' * 30)
    print("The number of scans in the dataset is: {:,}".format(len(scans_list)))
    print("The number of scans in the training set is: {:,}".format(len(train_scans_list)))
    print("The number of scans in the validation set is: {:,}".format(len(val_scans_list)))
    print('=' * 30)
          
    train_dataset = Dataset(
        scans_list = train_scans_list,
        scans_dir = os.path.join('/content/data_dir', config['dataset'], 'scans'),
        sv_bp_dir = os.path.join('/content/data_dir', config['dataset'], 'sv_bp'),
        scan_ext = config['scan_ext'],
        sv_bp_ext = config['singl_v_bp_ext'],
        num_of_views = config['input_channels'])
    
    val_dataset = Dataset(
        scans_list = val_scans_list,
        scans_dir = os.path.join('/content/data_dir', config['dataset'], 'scans'),
        sv_bp_dir = os.path.join('/content/data_dir', config['dataset'], 'sv_bp'),
        scan_ext = config['scan_ext'],
        sv_bp_ext = config['singl_v_bp_ext'],
        num_of_views = config['input_channels'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    experiment_log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('MSE', []),
        ('val_loss', []),
        ('val_MSE', []),
    ])

    
    trigger = 0
    
    for epoch in range(config['epochs']):
        print('====== Current Epoch is: %d / %d' % (epoch, config['epochs']))

        # Training (single EPOCH complete training)
        train_log = train(config, train_loader, model, criterion, optimizer)
        
        # Validation (single EPOCH complete validation)
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        print('Training Metrics: loss %.4f - MSE %.4f'
              % (train_log['loss'], train_log['MSE']))
        print('Validation Metrics: val_loss %.4f - val_MSE %.4f'
              % (val_log['loss'], val_log['MSE']))
        
        if scheduler is not None:
          current_lr = scheduler.get_last_lr()
          
        else:
          current_lr = config['lr']
          
          
        experiment_log['epoch'].append(epoch)
        experiment_log['lr'].append(current_lr)
        experiment_log['loss'].append(train_log['loss'])
        experiment_log['MSE'].append(train_log['MSE'])
        experiment_log['val_loss'].append(val_log['loss'])
        experiment_log['val_MSE'].append(val_log['MSE'])

        pd.DataFrame(experiment_log).to_csv('experiments/%s/experiment_log.csv' %
                                 config['experiment_name'], index=False)

        trigger += 1

        if epoch == 0:
            best_MSE = val_log['MSE'] + val_log['MSE']/10

        if val_log['MSE'] < best_MSE:
            torch.save(model.state_dict(), 'experiments/%s/best_model.pth' %
                       config['experiment_name'])
            best_MSE = val_log['MSE']
            print("##====## Saved Best Model ##====##")
            trigger = 0

        # Creat A new line to seperate output prints from each EPOCH
        print()


        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
