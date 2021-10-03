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
from utils.point_losses_LVOT import DSNTDoubleLoss

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

sys.path.insert(0, '..')
from networks.resnet50_torchvision import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
from networks.unet import UNet
import segmentation_models_pytorch as smp

from itertools import product 
from datetime import datetime

import matplotlib.pyplot as plt


def show_preds_heatmap(preds):
    ''' creates heatmaps and orders them for saving in summary_writer '''
    preds_detached = preds.detach().cpu().numpy()

    stack_np = []
    for layer in preds_detached:
        for channel in layer:
            maxval = np.max(channel)
            minval = np.min(channel)
            temp = (channel - minval) / (maxval - minval)
            
            cmap = plt.get_cmap('jet')
            rgba_img = cmap(temp.squeeze())
            rgb_img = rgba_img[:, :, :-1] # deletes alphachannel
            plot_tb = (rgb_img * 255).astype(np.uint8).transpose((2, 0, 1))
            stack_np.append(plot_tb)

            #plot = Image.fromarray((rgb_img * 255).astype(np.uint8))
            #plot.show()
    
    plot_np = np.stack(stack_np)
    
    return plot_np


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
              n_channels=3,
              epochs=30,
              learning_rate=0.001,
              batch_size=10,
              batch_accumulation=1,
              img_scale=1,
              transfer_learning_path=''):

    ''' define optimizer and loss '''
    #optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)
    criterion = DSNTDoubleLoss()

    ''' to check if training is from scratch or transfer learning/checkpoint appending '''
    if transfer_learning_path != '':
        try:
            ''' loads model parameters and optimizer status if logged '''
            start_checkpoint = torch.load(transfer_learning_path)
            net.load_state_dict(start_checkpoint['model_state_dict'])
            optimizer.load_state_dict(start_checkpoint['optimizer_state_dict'])
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
    train = BasicDataset(train_imgs_dir, train_masks_dir, img_scale=img_scale)
    n_train = len(train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    if data_train_and_validation[1] != '':
        val = BasicDataset(validate_imgs_dir, validate_masks_dir, img_scale)
        n_val = len(val)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    ''' make summary writer file with timestamp '''
    time_stamp = datetime.now().strftime('%b%d_%H-%M-%S')
    #hostname = socket.gethostname()
    if data_train_and_validation[1] == '':
        train_and_val = f'T-{data_train_and_validation[0]}_V-NONE'
    else:
        train_and_val = f'T-{data_train_and_validation[0]}_V-{data_train_and_validation[1]}'
    true_batch_size = batch_size * batch_accumulation
    writer = SummaryWriter(path.join(summary_writer_dir, f'{time_stamp}_{model_name}_{train_and_val}_{train_type}{epochs}_LR{learning_rate}_BS{true_batch_size}_SCL{img_scale}'))

    if data_train_and_validation[1] != '':
        logging.info(f'''Starting training:
            Training with:      {data_train_and_validation[0]}
            Training size:      {n_train}
            Validating with:    {data_train_and_validation[1]}
            Validation size:    {n_val}
            Epochs:             {epochs}
            Batch size:         {batch_size} x {batch_accumulation}
            Learning rate:      {learning_rate}
            Device:             {device.type}
            Images scaling:     {img_scale}
            Transfer learning:  {transfer_learning_path}
        ''')
    else:
        logging.info(f'''Starting training:
            Training with:      {data_train_and_validation[0]}
            Training size:      {n_train}
            Validation with:    {data_train_and_validation[1]}
            Epochs:             {epochs}
            Batch size:         {batch_size} x {batch_accumulation}
            Learning rate:      {learning_rate}
            Device:             {device.type}
            Images scaling:     {img_scale}
            Transfer learning:  {transfer_learning_path}
        ''')
    
    global_step = 0
    
    for epoch in range(epochs):
        net.train()

        with tqdm(total=n_train, desc=f'Epoch {start_epoch + epoch + 1}/{start_epoch + epochs}', unit='img') as pbar:
        
            optimizer.zero_grad()
            loss_batch = 0
            
            for i, batch in enumerate(train_loader):
                imgs = batch['image']
                true_masks = [batch['mask_i'], batch['mask_s']]
                assert imgs.shape[1] == n_channels, \
                    f'Network has been defined with {n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 #if net.output_channels == 1 else torch.long
                
                true_masks_cat = torch.cat((true_masks[0], true_masks[1]), 1).to(device=device, dtype=mask_type)

                preds = net(imgs)
                #preds = preds['out'] #torchvision syntax

                loss = criterion(preds, true_masks_cat)
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
                    if global_step % ((n_train / 10) // true_batch_size) == 0 and data_train_and_validation[1] != '':
                        
                        ''' show predictions in heatmap format '''
                        preds_heatmap = show_preds_heatmap(preds)
                        
                        for tag, value in net.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                        val_i_mean, val_s_mean, val_tot_mean, val_tot_median, val_diam_mean, val_diam_median = validate_mean_and_median_for_distance_and_diameter(
                            net, val_loader, device)
                        #writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                        #logging.info('Validation ED inferior mean: {}'.format(val_i_mean))
                        writer.add_scalar('ED_inferior_mean/eval', val_i_mean, global_step)

                        #logging.info('Validation ED superior mean: {}'.format(val_s_mean))
                        writer.add_scalar('ED_superior_mean/eval', val_s_mean, global_step)

                        logging.info('Validation ED total pixel mean: {}'.format(val_tot_mean))
                        writer.add_scalar('ED_total_mean/eval', val_tot_mean, global_step)
                        logging.info('Validation ED total pixel median: {}'.format(val_tot_median))
                        writer.add_scalar('ED_total_median/eval', val_tot_median, global_step)

                        logging.info('Validation LVOT diameter pixel mean: {}'.format(val_diam_mean))
                        writer.add_scalar('LVOT_diameter_pixel_mean/eval', val_diam_mean, global_step)
                        logging.info('Validation LVOT diameter pixel median: {}'.format(val_diam_median))
                        writer.add_scalar('LVOT_diameter_pixel_median/eval', val_diam_median, global_step)

                        #writer.add_images('images', imgs, global_step)
                        #writer.add_images('masks/true', true_masks_cat.view(true_masks_cat.shape[0] * true_masks_cat.shape[1], 1, true_masks_cat.shape[2], true_masks_cat.shape[3]), global_step)
                        #writer.add_images('masks/pred', preds_heatmap, global_step)
                    
                    optimizer.zero_grad()
                    
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

        ''' save model state without optimizer '''
        #torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}_LR_{learning_rate}_BS_{batch_size}_SCALE_{img_scale}_VAL_{val_percent}_RUNTIME_{timestamp}.pth')

    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    data_train_dir = path.join('data', 'train')
    data_validate_dir = path.join('data', 'validate')
    checkpoints_dir = 'checkpoints'
    summary_writer_dir = 'runs'
    
    ''' define model_name before running '''
    model_name = 'EFFIB0-DLV3+_DSNT_ADAM'
    n_classes = 2
    n_channels = 1
    
    training_parameters = dict(
        data_train_and_validation = [
            ['AVA1314X5_HMHM_K1', 'AVA1314X5_HMHM_K1'],
            ['AVA1314X5_HMHM_K2', 'AVA1314X5_HMHM_K2'],
            ['AVA1314X5_HMHM_K3', 'AVA1314X5_HMHM_K3'],
            ['AVA1314X5_HMHM_K4', 'AVA1314X5_HMHM_K4'],
            ['AVA1314X5_HMHM_K5', 'AVA1314X5_HMHM_K5']
            ],
        epochs=[30],
        learning_rate=[0.001],
        batch_size=[10],
        batch_accumulation=[2],
        img_scale=[1],
        transfer_learning_path=['']
    )
    
    ''' used to train multiple models in succession. add variables to arrays to make more combinations '''
    param_values = [v for v in training_parameters.values()]
    for data_train_and_validation, epochs, learning_rate, batch_size, batch_accumulation, img_scale, transfer_learning_path in product(*param_values):

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

        #net = fcn_resnet50(pretrained=False, progress=True, in_channels=n_channels, num_classes=n_classes, aux_loss=None)
        net = smp.DeepLabV3Plus(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=n_channels, classes=n_classes)

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
                      n_channels=n_channels,
                      epochs=epochs,
                      learning_rate=learning_rate,
                      batch_size=batch_size,
                      batch_accumulation=batch_accumulation,
                      img_scale=img_scale,
                      transfer_learning_path=transfer_learning_path)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)