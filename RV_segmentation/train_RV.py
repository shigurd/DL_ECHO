import logging
import os
import os.path as path
import sys

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from LV_segmentation.utils.validation_LV import validate_mean_and_median, validate_mean_and_median_hd
from LV_segmentation.utils.dataloader_LV import BasicDataset
from LV_segmentation.utils.segmentation_losses_LV import DiceSoftBCELoss, IoUSoftBCELoss, DiceSoftLoss, IoUSoftLoss

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

sys.path.insert(0, '..')
from networks.resnet50_torchvision import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
from networks.unet import UNet
from networks.unet_plusplus import NestedUNet
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
              mid_systole_only=False,
              with_augmentations=False,
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)  # patience is number of epochs for improvement

    if loss_function == 'DICE':
        criterion = DiceSoftLoss()
    elif loss_function == 'DICEBCE':
        criterion = DiceSoftBCELoss()
    elif loss_function == 'IOU':
        criterion = IoUSoftLoss()
    elif loss_function == 'IOUBCE':
        criterion = IoUSoftBCELoss()


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
        train_type = f'TRANSFER-EP{start_epoch}+'
    else:
        start_epoch = 0
        train_type = 'EP'
    
    ''' dataloader for training and evaluation '''
    train = BasicDataset(train_imgs_dir, train_masks_dir, img_scale=img_scale, mid_systole_only=mid_systole_only, coord_conv=with_cc)
    n_train = len(train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    if data_train_and_validation[1] != '':
        val = BasicDataset(validate_imgs_dir, validate_masks_dir, img_scale=img_scale, mid_systole_only=mid_systole_only, coord_conv=with_cc)
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
        other_additions += '_TF'

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
            Mid systole only     {mid_systole_only}
            With augmentations:  {with_augmentations}
            With cc:             {with_cc}
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
                true_masks = batch['mask']
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
                #preds = preds['out'] # use for torchvision networks

                loss = criterion(preds, true_masks)
                loss_batch += loss.item() # compensates for batch repeat
                
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

                    # validates every 10% of the epoch
                    #if global_step % ((n_train / 1) // true_batch_size) == 0 and data_train_and_validation[1] != '':

                    optimizer.zero_grad()

        if data_train_and_validation[1] != '':
            ''' logging moved here for epoch '''
            #for tag, value in net.named_parameters():
                # tag = tag.replace('.', '/')
                # writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
            validate_mean_dice, validate_median_dice, validate_mean_iou, validate_median_iou, validate_mean_hd, validate_median_hd = validate_mean_and_median_hd(net, val_loader, device)

            if lr_decay:
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning_rate', current_lr, global_step)
                logging.info('Learning rate : {}'.format(current_lr))

                ''' for some reason efficientnets always gets a learning breakthough at 7000 images seen '''
                if true_batch_size * global_step >= 7000:
                    scheduler.step(validate_mean_dice)

            logging.info('Validation Mean Dice: {}'.format(validate_mean_dice))
            logging.info('Validation Median Dice: {}'.format(validate_median_dice))
            writer.add_scalar('Mean_Dice/eval', validate_mean_dice, global_step)
            writer.add_scalar('Median_Dice/eval', validate_median_dice, global_step)

            #logging.info('Validation Mean IoU: {}'.format(validate_mean_iou))
            #logging.info('Validation Median IoU: {}'.format(validate_median_iou))
            #writer.add_scalar('Mean_IoU/eval', validate_mean_iou, global_step)
            #writer.add_scalar('Median_IoU/eval', validate_median_iou, global_step)

            logging.info('Validation Mean HD: {}'.format(validate_mean_hd))
            logging.info('Validation Median HD: {}'.format(validate_median_hd))
            writer.add_scalar('Mean_HD/eval', validate_mean_hd, global_step)
            writer.add_scalar('Median_HD/eval', validate_median_hd, global_step)

            if log_heatmaps == True:
                ''' show prediction in heatmap format '''
                preds_heatmap = show_preds_heatmap(preds)
                writer.add_images('images', imgs, global_step)
                writer.add_images('masks/true', true_masks, global_step)
                writer.add_images('masks/pred', torch.sigmoid(preds.detach().cpu()) > 0.5, global_step)
                writer.add_images('heatmap/pred', preds_heatmap, global_step)

        else:
            ''' lower learning rate for optim for 2 stages '''
            if epoch == 18:
                optimizer.param_groups[0]['lr'] *= 0.1
            elif epoch == 23:
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
                }, path.join(checkpoints_dir, f'{time_stamp}_{model_name}_{train_and_val}_{train_type}{epoch + 1}_LR{learning_rate}_BS{true_batch_size}_SCL{img_scale}.pth'))
            logging.info(f'Checkpoint {start_epoch + epoch + 1} saved !')

            logs_file = open(logs_pth, 'a')
            logs_file.writelines(
                f'{run_name},{validate_mean_dice},{validate_median_dice},{validate_mean_hd},{validate_median_hd}\n')
            logs_file.close()

            ''' add hyperparams to summarywriter for filtering '''
            writer.add_hparams(
                {'model': model_name, 'optim': optimizer_function, 'loss_func': loss_function, 'bs': true_batch_size,
                 'lr': learning_rate, 'aug': with_augmentations},
                {'hparams/val_mean_dice': validate_mean_dice, 'hparams/val_mean_hd': validate_mean_hd})

    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    data_train_dir = path.join('data', 'train')
    data_validate_dir = path.join('data', 'validate')
    checkpoints_dir = 'checkpoints'
    summary_writer_dir = 'runs'
    logs_pth = 'runs/logs.csv'

    n_classes = 1
    n_channels = 1

    training_parameters = dict(
        model_name=['EFFIB2UNET', 'EFFIB1UNET', 'RES50UNET'],
        loss_function=['DICEBCE'],
        optimizer_function=['ADAM', 'RMSP'],
        epochs=[30],
        learning_rate=[0.01, 0.005, 0.001, 0.0005],
        batch_size=[8],
        batch_accumulation=[1, 2, 4, 8],
        img_scale=[1],
        mid_systole_only=[True],
        with_augmentations=[True],
        with_cc=[False],
        lr_decay=[True],
        transfer_learning_path=[''],
        log_heatmaps=[False],
        data_train_and_validation=[
            ['RV141_HMLHML_K1', 'RV141_HMLHML_K1'],
        ]
    )
    
    ''' used to train multiple models in succession. add variables to arrays to make more combinations '''
    param_values = [v for v in training_parameters.values()]
    for model_name, loss_function, optimizer_function, epochs, learning_rate, batch_size, batch_accumulation, img_scale, mid_systole_only, with_augmentations, with_cc, lr_decay, transfer_learning_path, log_heatmaps, data_train_and_validation in product(*param_values):

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
            net = smp.Unet(encoder_name="efficientnet-b2", encoder_weights=None, in_channels=n_channels,
                           classes=n_classes)
        elif model_name == 'EFFIB2UNETIMGN':
            net = smp.Unet(encoder_name="efficientnet-b2", encoder_weights='imagenet', in_channels=n_channels,
                           classes=n_classes)
        elif model_name == 'RES50UNET':
            net = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=n_channels, classes=n_classes)
        elif model_name == 'RES50UNETIMGN':
            net = smp.Unet(encoder_name="resnet50", encoder_weights='imagenet', in_channels=n_channels,
                           classes=n_classes)
        elif model_name == 'EFFIB1UNET':
            net = smp.Unet(encoder_name="efficientnet-b1", encoder_weights=None, in_channels=n_channels,
                           classes=n_classes)
        elif model_name == 'UNET':
            net = UNet(n_channels, n_classes, bilinear=False)


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
                      mid_systole_only=mid_systole_only,
                      with_augmentations=with_augmentations,
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