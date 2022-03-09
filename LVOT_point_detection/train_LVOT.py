import logging
import os
import os.path as path
import sys

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from utils.validation_LVOT import validate_mean_and_median_for_distance_and_diameter
from utils.dataloader_LVOT import BasicDataset
from utils.point_losses_LVOT import DSNTDistanceDoubleLossNew, DSNTDoubleLossNew, DSNTDoubleLossNewMSEC, DSNTJSDDoubleLossNew, DSNTCosDoubleLossNew

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.insert(0, '..')
from networks.resnet50_torchvision import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
from networks.unet import UNet
from common_utils.heatmap_plot import show_preds_heatmap
from common_utils.live_augmentation_skimage import augment_imgs_masks_batch
import segmentation_models_pytorch as smp

from itertools import product 
from datetime import datetime


def train_net(net,
              device,
              model_name,
              data_train_and_validation,
              checkpoints_dir,
              summary_writer_dir,
              train_imgs_dir,
              train_masks_dir,
              validate_imgs_dir,
              validate_masks_dir,
              loss_function='DSNT',
              optimizer_function='ADAM',
              n_channels=1,
              epochs=30,
              learning_rate=0.001,
              batch_size=10,
              batch_accumulation=1,
              img_scale=1,
              with_augmentations=False,
              with_gaussian=False,
              with_cc=False,
              lr_decay=False,
              transfer_learning_path='',
              log_heatmaps=False,
              logs_pth='runs/logs.csv'):

    ''' define optimizer function '''
    if optimizer_function == 'ADAM':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    elif optimizer_function == 'RMSP':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    elif optimizer_function == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-8)

    if lr_decay == True:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.01)  # patience is number of epochs for improvement

    ''' define loss function '''
    if loss_function == 'DSNT':
        criterion = DSNTDoubleLossNew()
    elif loss_function == 'DSNTMSEC':
        criterion = DSNTDoubleLossNewMSEC()
    elif loss_function == 'DSNTJSD':
        #assert with_gaussian == True
        criterion = DSNTJSDDoubleLossNew()
    elif loss_function == 'DSNTDIST':
        criterion = DSNTDistanceDoubleLossNew()
    elif loss_function == 'DSNTCOS':
        criterion = DSNTCosDoubleLossNew()

    ''' to check if training is from scratch or transfer learning/checkpoint appending '''
    if transfer_learning_path != '':
        try:
            ''' loads model parameters and optimizer status if logged '''
            start_checkpoint = torch.load(transfer_learning_path)
            net.load_state_dict(start_checkpoint['model_state_dict'])
            #optimizer.load_state_dict(start_checkpoint['optimizer_state_dict'])
            start_epoch = start_checkpoint['epoch']
        except:
            ''' loads only model parameters '''
            net.load_state_dict(torch.load(transfer_learning_path, map_location=device))
            start_epoch = 0
        train_type = f'TF-EP{start_epoch}+'
    else:
        start_epoch = 0
        train_type = 'EP'
    
    ''' dataloader for training and evaluation '''
    train = BasicDataset(train_imgs_dir, train_masks_dir, img_scale=img_scale, with_gaussian=with_gaussian, with_cc=with_cc)
    n_train = len(train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    if data_train_and_validation[1] != '':
        val = BasicDataset(validate_imgs_dir, validate_masks_dir, img_scale)
        n_val = len(val)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    ''' make summary writer file with timestamp '''
    time_stamp = datetime.now().strftime('%b%d_%H-%M-%S')
    #hostname = socket.gethostname()
    if data_train_and_validation[1] == '':
        train_and_val = f'T-{data_train_and_validation[0]}_V-NONE'
    else:
        train_and_val = f'T-{data_train_and_validation[0]}_V-{data_train_and_validation[1]}'
    true_batch_size = batch_size * batch_accumulation

    other_additions = ''
    if lr_decay == True:
        other_additions += '_LR5'
    if with_augmentations == True:
        other_additions += '_AL'
    if with_cc == True:
        assert n_channels == 3
        other_additions += '_CC'
    if transfer_learning_path != '':
        other_additions += f'_TF-{transfer_learning_path.rsplit("T-", 1)[-1].rsplit("_", 4)[0]}'

    run_name = f'{time_stamp}_{model_name}_{loss_function}_{optimizer_function}{other_additions}_{train_and_val}_{train_type}{epochs}_LR{learning_rate}_BS{true_batch_size}'
    writer = SummaryWriter(path.join(summary_writer_dir, run_name))

    logging.info(f'''Starting training:
        Device:              {device.type}
        Model:               {model_name}
        Loss function:       {loss_function}
        Optimizer:           {optimizer_function}
        Training data:       {data_train_and_validation[0]}  
        Validation data:     {data_train_and_validation[1]} 
        Epochs:              {epochs}
        Batch size:          {batch_size} x {batch_accumulation}
        Learning rate:       {learning_rate}
        LR decay:            {lr_decay}
        Images scaling:      {img_scale}
        With augmentations:  {with_augmentations}
        With cc:             {with_cc}
        With gaussian:       {with_gaussian}
        Transfer learning:   {transfer_learning_path}
        Log heatmaps:        {log_heatmaps}
    ''')
    
    global_step = 0
    
    for epoch in range(epochs):
        net.train()

        with tqdm(total=n_train, desc=f'Epoch {start_epoch + epoch + 1}/{start_epoch + epochs}', unit='img') as pbar:
        
            optimizer.zero_grad()
            loss_batch = 0
            
            for i, batch in enumerate(train_loader):
                imgs = batch['image']
                true_masks = torch.cat((batch['mask_i'], batch['mask_s']), 1)
                assert imgs.shape[1] == n_channels, \
                    f'Network has been defined with {n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                if with_augmentations == True:
                    ''' apply augmentations to images and masks '''
                    imgs_augmented_batch, masks_augmented_batch = augment_imgs_masks_batch(imgs, true_masks)
                    imgs = imgs_augmented_batch.to(device=device, dtype=torch.float32)
                    true_masks = masks_augmented_batch.to(device=device, dtype=torch.float32)
                else:
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)

                preds = net(imgs)
                #preds = preds['out'] #torchvision syntax

                loss = criterion(preds, true_masks)
                loss_batch += loss.item() #moved to compensate for batch repeat
                
                loss.backward()
                #nn.utils.clip_grad_value_(net.parameters(), 0.1)
                
                ''' only update optimizer after accumulation '''
                if (i + 1) % batch_accumulation == 0:
                    writer.add_scalar('Loss/train', loss_batch / batch_accumulation, global_step)
                    pbar.set_postfix(**{'loss (batch)': loss_batch / batch_accumulation})
                    optimizer.step()
                    pbar.update(imgs.shape[0] * batch_accumulation)
                    
                    global_step += 1
                    loss_batch = 0
                    
                    ''' validates every 10% of the epoch '''
                    #if global_step % ((n_train / (1/2)) // true_batch_size) == 0 and data_train_and_validation[1] != '':
                    
                    optimizer.zero_grad()

        if data_train_and_validation[1] != '':
            ''' logging moved here for epoch '''
            #for tag, value in net.named_parameters():
                #tag = tag.replace('.', '/')
                # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            val_i_mean, val_s_mean, val_tot_mean, val_tot_median, val_diam_mean, val_diam_median, val_x_tot_mean, val_y_tot_mean = validate_mean_and_median_for_distance_and_diameter(
                net, val_loader, device)

            if lr_decay:
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning_rate', current_lr, global_step)
                logging.info('Learning rate : {}'.format(current_lr))

                ''' for some reason efficientnets always gets a learning breakthough at 7000 images seen '''
                if true_batch_size * global_step >= 7000:
                    scheduler.step(val_tot_mean)

            # logging.info('Validation ED inferior mean: {}'.format(val_i_mean))
            writer.add_scalar('ED_inferior_mean/eval', val_i_mean, global_step)
            # logging.info('Validation ED superior mean: {}'.format(val_s_mean))
            writer.add_scalar('ED_superior_mean/eval', val_s_mean, global_step)

            # logging.info('Validation tot x diff  mean: {}'.format(val_x_tot_mean))
            writer.add_scalar('x_tot_mean/eval', val_x_tot_mean, global_step)
            # logging.info('Validation tot y diff mean: {}'.format(val_y_tot_mean))
            writer.add_scalar('y_tot_mean/eval', val_y_tot_mean, global_step)

            logging.info('Validation ED total pixel mean: {}'.format(val_tot_mean))
            writer.add_scalar('ED_total_mean/eval', val_tot_mean, global_step)
            logging.info('Validation ED total pixel median: {}'.format(val_tot_median))
            writer.add_scalar('ED_total_median/eval', val_tot_median, global_step)

            logging.info('Validation LVOT diameter pixel mean: {}'.format(val_diam_mean))
            writer.add_scalar('LVOT_diameter_pixel_mean/eval', val_diam_mean, global_step)
            logging.info('Validation LVOT diameter pixel median: {}'.format(val_diam_median))
            writer.add_scalar('LVOT_diameter_pixel_median/eval', val_diam_median, global_step)

            if log_heatmaps == True:
                ''' show predictions in heatmap format '''
                preds_heatmap = show_preds_heatmap(preds)
                writer.add_images('images', imgs, global_step)
                writer.add_images('masks/true', true_masks.view(true_masks.shape[0] * true_masks.shape[1], 1, true_masks.shape[2], true_masks.shape[3]), global_step)
                writer.add_images('masks/pred', preds_heatmap, global_step)
        else:
            if lr_decay == True:
                ''' maunually defined lr lowering '''
                if epoch == 20:
                    optimizer.param_groups[0]['lr'] *= 0.1
                elif epoch == 28:
                    optimizer.param_groups[0]['lr'] *= 0.1
                current_lr = optimizer.param_groups[0]['lr']
                logging.info('Learning rate : {}'.format(current_lr))

        try:
            os.mkdir(checkpoints_dir)
            logging.info('Created checkpoint directory')
        except OSError:
            pass

        ''' create checkpoint at end of epochs '''
        if epoch + 1 == epochs: 
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path.join(checkpoints_dir, f'{run_name}.pth'))
            logging.info(f'Checkpoint {start_epoch + epoch + 1} saved !')

            logs_file = open(logs_pth, 'a')
            if data_train_and_validation[1] == '':
                logs_file.writelines(f'{run_name},NONE,NONE,NONE,NONE,NONE,NONE\n')
            else:
                logs_file.writelines(f'{run_name},{val_tot_mean},{val_tot_median},{val_diam_mean},{val_diam_median},{val_x_tot_mean},{val_y_tot_mean}\n')
                ''' add hyperparams to summarywriter for filtering '''
                writer.add_hparams({'model': model_name, 'optim': optimizer_function, 'loss_func': loss_function,
                                    'bs': true_batch_size, 'lr': learning_rate, 'aug': with_augmentations},
                                   {'hparams/val_tot_mean': val_tot_mean, 'hparams/val_diam_mean': val_diam_mean})

            logs_file.close()


    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    data_train_dir = path.join('data', 'train')
    data_validate_dir = path.join('data', 'validate')
    checkpoints_dir = 'checkpoints'
    summary_writer_dir = 'runs'
    logs_pth = 'runs/logs.csv'

    n_classes = 2
    n_channels = 1
    
    training_parameters = dict(
        model_name=['EFFIB2UNET'],
        loss_function=['DSNT'],
        optimizer_function=['ADAM'],
        epochs=[30],
        learning_rate=[0.003],
        batch_size=[8],
        batch_accumulation=[4],
        img_scale=[1],
        with_augmentations=[True],
        with_gaussian=[False],
        with_cc=[False],
        lr_decay=[True],
        transfer_learning_path=['checkpoints/Mar08_23-28-45_EFFIB2UNETIMGN_DSNT_ADAM_LR5_AL_T-CAMUS1800MVROT_HML_V-NONE_EP30_LR0.003_BS32.pth'],
        log_heatmaps=[False],
        data_train_and_validation=[

            ['GE1408_HMLHMLAVA_K2', 'GE1408_HMHMAVA_K2'],
            ['GE1408_HMLHMLAVA_K3', 'GE1408_HMHMAVA_K3'],
            ['GE1408_HMLHMLAVA_K4', 'GE1408_HMHMAVA_K4'],
            ['GE1408_HMLHMLAVA_K5', 'GE1408_HMHMAVA_K5']
        ]
    )
    
    ''' used to train multiple models in succession. add variables to arrays to make more combinations '''
    param_values = [v for v in training_parameters.values()]
    for model_name, loss_function, optimizer_function, epochs, learning_rate, batch_size, batch_accumulation, img_scale, with_augmentations, with_gaussian, with_cc, lr_decay, transfer_learning_path, log_heatmaps, data_train_and_validation in product(*param_values):

        current_train_imgs_dir = path.join(data_train_dir, 'imgs', data_train_and_validation[0])
        current_train_masks_dir = path.join(data_train_dir, 'masks', data_train_and_validation[0])
        
        if data_train_and_validation[1] != '':
            current_validate_imgs_dir = path.join(data_validate_dir, 'imgs', data_train_and_validation[1])
            current_validate_masks_dir = path.join(data_validate_dir, 'masks', data_train_and_validation[1])
        else:
            current_validate_imgs_dir = ''
            current_validate_masks_dir = ''
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {device}')

        if model_name == 'EFFIB2UNET':
            net = smp.Unet(encoder_name="efficientnet-b2", encoder_weights=None, in_channels=n_channels, classes=n_classes)
        elif model_name == 'EFFIB2UNETIMGN':
            net = smp.Unet(encoder_name="efficientnet-b2", encoder_weights='imagenet', in_channels=n_channels, classes=n_classes)
        elif model_name == 'RES50UNET':
            net = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=n_channels, classes=n_classes)
        elif model_name == 'RES50UNETIMGN':
            net = smp.Unet(encoder_name="resnet50", encoder_weights='imagenet', in_channels=n_channels, classes=n_classes)
        elif model_name == 'UNET':
            net = UNet(n_channels, n_classes, bilinear=False)
        elif model_name == 'RES50FCN':
            net = fcn_resnet50(pretrained=False, progress=True, in_channels=n_channels, num_classes=n_classes, aux_loss=None)
        elif model_name == 'RES50FCNCOCO':
            assert n_channels == 3
            net = fcn_resnet50(pretrained=True, progress=True, in_channels=n_channels, num_classes=21, aux_loss=None) #num_classes = 21 is necessary to load the pretrained model
            net.fc = nn.Linear(512, n_classes)  # to change final pretrained output layer for torchvision models

        logging.info(f'Network:\n'
                     f'\t{n_channels} input channels\n'
                     f'\t{n_classes} output channels\n')

        net.to(device=device)
        ''' faster convolutions, but more memory '''
        torch.backends.cudnn.benchmark = True

        try:
            train_net(net=net,
                      device=device,
                      model_name=model_name,
                      data_train_and_validation=data_train_and_validation,
                      checkpoints_dir=checkpoints_dir,
                      summary_writer_dir=summary_writer_dir,
                      train_imgs_dir=current_train_imgs_dir,
                      train_masks_dir=current_train_masks_dir,
                      validate_imgs_dir=current_validate_imgs_dir,
                      validate_masks_dir=current_validate_masks_dir,
                      loss_function=loss_function,
                      optimizer_function=optimizer_function,
                      n_channels=n_channels,
                      epochs=epochs,
                      learning_rate=learning_rate,
                      batch_size=batch_size,
                      batch_accumulation=batch_accumulation,
                      img_scale=img_scale,
                      with_augmentations=with_augmentations,
                      with_gaussian=with_gaussian,
                      with_cc=with_cc,
                      lr_decay=lr_decay,
                      transfer_learning_path=transfer_learning_path,
                      log_heatmaps=log_heatmaps,
                      logs_pth=logs_pth)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
