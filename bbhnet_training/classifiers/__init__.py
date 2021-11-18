
import logging

from . import cnn_small
from . import cnn_medium
from . import cnn_large
from . import fc_corr

ALL_NAMES = {
    'CNN-SMALL': cnn_small,
    'CNN-MEDIUM': cnn_medium,
    'CNN-LARGE': cnn_large,
    'FC-CORR': fc_corr,
}

def get_arch(name):
    logging.info('Choose {} as architecture'.format(name))
    return ALL_NAMES[name]
