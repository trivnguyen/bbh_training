#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import logging

# set up logger
logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO, stream=sys.stdout)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from gwpy.timeseries import TimeSeries

# import customized training package
from bbhnet_training import classifiers, utils, data_utils, training_logger


# parse command-line arguments
def parse_cmd():
    parser = argparse.ArgumentParser()

    # io arguments
    parser.add_argument('-i', '--input-dir', required=True,
                        help='Path to input dataset directory')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Path to output directory')
    parser.add_argument('--store-val-output', action='store_true',
                        help='Enable to store validation data NN output to outdir')

    # nn architecture args
    parser.add_argument(
        '-A', '--arch', choices=(classifiers.ALL_NAMES.keys()), type=str.upper, required=True,
        help='NN architecture to use')
    parser.add_argument(
        '--input-shape', nargs='+', type=int, default=(2, 1024), required=False,
        help='Input dimension for NN, excluding batch dimension. Default is (2, 1024)')
    parser.add_argument(
        '--corr-dim', type=int, default=80, required=False,
        help='Dimension of correlation array. If 0 (default) is given, correlation will not be used')

    # training args
    parser.add_argument(
        '-e', '--max-epochs', type=int, default=30, required=False,
        help='Maximum number of epochs')
    parser.add_argument(
        '-b','--batch-size', type=int, default=1024, required=False,
        help='Batch size')
    parser.add_argument(
        '-lr', '--learning-rate', type=float, default=1e-3, required=False,
        help='Learning rate of ADAM optimizer')
    parser.add_argument(
        '-l1', '--weight-decay', type=float, default=1e-3, required=False,
        help='L1 regularization of ADAM optimizer')
    parser.add_argument('--shuffle', action='store_true',
                        help='Enable to shuffle training data')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training using the latest model available')

    return parser.parse_args()


if __name__ == '__main__':

    # parse command-line arguments
    FLAGS = parse_cmd()

    # get CPU/GPU device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
    else:
        device = torch.device('cpu')
        pin_memory = False

    # initialize Dataset and DataLoader objects
    # use correlation if dimension is given
    if FLAGS.corr_dim > 0:
        keys = ('data', 'label', 'corr')
    else:
        keys = ('data', 'label')
    train_dataset = data_utils.Dataset(
        os.path.join(FLAGS.input_dir, 'train'), keys, FLAGS.shuffle)
    val_dataset = data_utils.Dataset(
        os.path.join(FLAGS.input_dir, 'val'), keys)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                              num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size,
                            num_workers=4, pin_memory=pin_memory)

    # initialize NN model
    arch = classifiers.get_arch(FLAGS.arch)
    net = arch.Classifier(input_shape=FLAGS.input_shape,
                          corr_dim=FLAGS.corr_dim)
    net = net.to(device)
    if FLAGS.resume:
        state = utils.get_checkpoint(os.path.join(FLAGS.output_dir, 'models'))
        logging.info('Resume training from state: {}'.format(state))
        net.load_state_dict(torch.load(state, map_location=device))

    # initialize loss function, optimizer, LR scheduler, and training logger
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    train_logger = training_logger.TrainingLogger(outdir=FLAGS.output_dir, metrics=['loss'])

    # print out some infomation before training
    logging.info('Begin training')
    logging.info('Number of training samples   : {} '.format(len(train_dataset)))
    logging.info('Number of validation samples : {} '.format(len(val_dataset)))
    logging.info('Input shape                  : {} '.format(FLAGS.input_shape))
    logging.info('Correlation array dim        : {} '.format(FLAGS.corr_dim))
    logging.info('------------------------')

    # start training
    n_batch_total = len(train_loader)

    for epoch in range(FLAGS.max_epochs):
        # training
        net.train()  # switch to training mode
        train_loss = 0.
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()  # reset gradient

            # type conversion and move to GPU if needed
            if FLAGS.corr_dim > 0:
                xb, yb, xcorrb = batch
                xb = xb.float().to(device)
                yb = yb.float().to(device)
                xcorrb = xcorrb.float().to(device)
            else:
                xb, yb = batch
                xb = xb.float().to(device)
                yb = yb.float().to(device)
                xcorrb = None

            # combined glitches and background label
            yb[yb == 2] = 0.


            # forward pass, calculate loss, backward pass, and GD
            yhatb = net(xb, xcorrb)
            loss = criterion(yhatb, yb)
            loss.backward()
            optimizer.step()

            # Update training loss for logging
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_dataset)

        # Update LR scheduler
        scheduler.step(train_loss)

        # validating
        net.eval()    # switch to validation mode
        val_loss = 0.
        with torch.no_grad():   # disable GD to save computation
            for i, batch in enumerate(val_loader):
                # type conversion and move to GPU if needed
                if FLAGS.corr_dim > 0:
                    xb, yb, xcorrb = batch
                    xb = xb.float().to(device)
                    yb = yb.float().to(device)
                    xcorrb = xcorrb.float().to(device)
                else:
                    xb, yb = batch
                    xb = xb.float().to(device)
                    yb = yb.float().to(device)
                    xcorrb = None

                # combine glitch and background label
                yb[yb == 2] = 0.

                # forward pass and calculate loss
                yhatb = net(xb, xcorrb)
                loss = criterion(yhatb, yb)

                # Update validation loss for logging
                val_loss += loss.item() * len(xb)

                # Also store validation output if enable
                if FLAGS.store_val_output:
                    train_log.update_predict(yhatb, yb)
        val_loss /= len(val_loader.dataset)

        # logging at the end of each epoch
        # store average loss per sample
        train_logger.update_metric('loss', train_loss, epoch, test_metric=val_loss,
                                n_batch_total=n_batch_total)

        # store metric and model
        train_logger.log_metric()
        train_logger.save_model(net, epoch)
        train_logger.save_optimizer(optimizer, epoch)

        # store val prediction
        if FLAGS.store_val_output:
            train_logger.save_predict(epoch)

        # print out status
        train_logger.display_status(
            'loss', train_loss, epoch, test_metric=val_loss,
            max_epochs=FLAGS.max_epochs, n_batch_total=n_batch_total)

