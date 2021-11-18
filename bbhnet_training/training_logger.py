
import os
import h5py
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch

# default plotting style
plt.style.use('seaborn-colorblind')
mpl.rc('font', size=15)
mpl.rc('figure', facecolor='w', figsize=(8, 5))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.INFO)


class TrainingLogger():
    ''' TrainingLogger keeps track of training progress: logging metrics,
    writing training checkpoint, etc.
    '''

    def __init__(self, outdir, metrics):
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.metrics = dict([(m, {'steps': [], 'epochs': [],
                                  'train': [], 'test': []}) for m in metrics])

        self.predict = []
        self.target = []

    def update_metric(self, metric_name, train_metric, epoch, test_metric=None,
                      n_batch_total=1, n_batch=None):
        # convert torch tensor to numpy array
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.data.cpu().numpy()
        if isinstance(test_metric, torch.Tensor):
            test_metric = test_metric.data.cpu().numpy()

        if test_metric is None:
            test_metric = np.nan

        # update metric to dictionary
        if n_batch is None:
            n_batch = n_batch_total
        step = self._step(epoch, n_batch, n_batch_total)
        self.metrics[metric_name]['train'].append(train_metric)
        self.metrics[metric_name]['test'].append(test_metric)
        self.metrics[metric_name]['steps'].append(step)
        self.metrics[metric_name]['epochs'].append(step / n_batch_total)

    def log_metric(self, metric_name=None, max_epochs=None):

        # create a directory for metric
        outdir = os.path.join(self.outdir, 'metrics')
        os.makedirs(outdir, exist_ok=True)

        # If name is not given, log all metrics
        if metric_name is not None:
            train = self.metrics[metric_name]['train']
            test = self.metrics[metric_name]['test']
            steps = self.metrics[metric_name]['steps']
            epochs = self.metrics[metric_name]['epochs']

            array = np.vstack((steps, epochs, train, test)).T
            header = 'Step     Epochs    Train     Test'
            np.savetxt(os.path.join(outdir, f'{metric_name}.txt'),
                       array, fmt=('%d, %.2f, %.5f, %.5f'), header=header)
            self._plot_metric(metric_name)
        else:
            for metric_name in self.metrics.keys():
                train = self.metrics[metric_name]['train']
                test = self.metrics[metric_name]['test']
                steps = self.metrics[metric_name]['steps']
                epochs = self.metrics[metric_name]['epochs']

                array = np.vstack((steps, epochs, train, test)).T
                header = 'Step     Epochs    Train     Test'
                np.savetxt(os.path.join(outdir, f'{metric_name}.txt'),
                           array, fmt=('%d, %.2f, %.5f, %.5f'), header=header)
                self._plot_metric(metric_name)

    def _plot_metric(self, metric_name):
            train = self.metrics[metric_name]['train']
            test = self.metrics[metric_name]['test']
            steps = self.metrics[metric_name]['steps']
            epochs = self.metrics[metric_name]['epochs']

            # plot
            fig, ax = plt.subplots()
            ax.plot(steps, train, label='Training')
            ax.plot(steps, test, label='Validation')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric_name)
            ax.legend()

            axtwin = ax.twiny()
            axtwin.plot(epochs, train, alpha=0.)
            axtwin.grid(False)
            axtwin.set_xlabel('Epoch')

            # save plot
            outdir = os.path.join(self.outdir, 'metrics')
            fig.savefig(os.path.join(outdir, f'{metric_name}.png'), dpi=300)
            plt.close()

    def display_status(self, metric_name, train_metric, epoch, test_metric=None,
                       max_epochs=None, n_batch_total=1, n_batch=None, show_epoch=True):
        # convert torch tensor to numpy array
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.data.cpu().numpy()
        if isinstance(test_metric, torch.Tensor):
            test_metric = test_metric.data.cpu().numpy()

        if n_batch is None:
            n_batch = n_batch_total
        if max_epochs is None:
            max_epochs = np.nan

        # print out epoch number
        if show_epoch:
            logging.info('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
                epoch, max_epochs, n_batch, n_batch_total))
        logging.info('Train {0:}: {1:.4f}, Test {0:}: {2:.4f}'.format(
            metric_name, train_metric, test_metric))

    def update_predict(self, predict, target):
        # convert torch tensor to numpy array
        if isinstance(predict, torch.Tensor):
            predict = predict.data.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.data.cpu().numpy()

        self.predict.append(predict)
        self.target.append(target)

    def save_predict(self, epoch, n_batch=None, reset=True):
        outdir = os.path.join(self.outdir, 'results')
        os.makedirs(outdir, exist_ok=True)
        if n_batch is not None:
            fn = os.path.join(outdir, f'epoch_{epoch}_batch_{n_batch}.h5')
        else:
            fn = os.path.join(outdir, f'epoch_{epoch}.h5')

        predict = np.concatenate(self.predict)
        target = np.concatenate(self.target)
        size = len(predict)

        # writing data in HDF5 format
        with h5py.File(fn, 'w') as f:
            f.attrs.update({
                'size': size,
                'epoch': epoch,
            })
            if n_batch is not None:
                f.attrs['n_batch'] = n_batch
            f.create_dataset('predict', data=predict, chunks=True)
            f.create_dataset('target', data=target, chunks=True)

        # reset array after storing data
        if reset:
            self.predict = []
            self.target = []

    def save_model(self, model, epoch, n_batch=None):
        outdir = os.path.join(self.outdir, 'models')
        os.makedirs(outdir, exist_ok=True)
        if n_batch is not None:
            torch.save(model.state_dict(),
                       os.path.join(outdir, f'epoch_{epoch}_batch_{n_batch}'))
        else:
            torch.save(model.state_dict(),
                       os.path.join(outdir, f'epoch_{epoch}'))

    def save_optimizer(self, optimizer, epoch, n_batch=None):
        outdir = os.path.join(self.outdir, 'optimizers')
        os.makedirs(outdir, exist_ok=True)
        if n_batch is not None:
            torch.save(optimizer.state_dict(),
                       os.path.join(outdir, f'epoch_{epoch}_batch_{n_batch}'))
        else:
            torch.save(optimizer.state_dict(),
                       os.path.join(outdir, f'epoch_{epoch}'))

    # Private Functionality
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

