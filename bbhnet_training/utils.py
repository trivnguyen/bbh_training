
import os
import sys
import glob
import logging

logger = logging.getLogger(__name__)

import numpy as np

def get_device(device):
    ''' Convenient func that sets up and prints out information about GPU '''
    if device.lower() == 'cpu':
        device = torch.device('cpu')
    elif 'cuda' in device.lower():
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            logging.warning('No GPU available. Use CPU instead.')
            device = torch.device('cpu')
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        total_memory *= 1e-9 # convert bytes to Gb
        logger.info('Use device: {}'.format(torch.cuda.get_device_name(device)))
        logger.info('Total memory: {:.4f} GB'.format(total_memory))
    else:
        logger.info('Use device: CPU')
    return device

def get_checkpoint(model_dir, epoch=None):
    ''' Get epoch from model directory. If not given, get latest '''
    if epoch is None:
        checkpoint = None
        for e in range(100000):
            temp = os.path.join(model_dir, f'epoch_{e}')
            if not os.path.exists(temp):
                return checkpoint
            checkpoint = temp
    else:
        return os.path.join(model_dir, f'epoch_{epoch}')

