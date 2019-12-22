import matplotlib
matplotlib.use('Agg')

import argparse, time, logging

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms

from gluoncv.data.sampler import SplitSampler

import horovod.mxnet as hvd

def main():
    hvd.init()
    a = mx.nd.array([4.0])
    print(a.asnumpy())
    hvd.allreduce_(a, name='a')
    print(a.asnumpy())
    hvd.allreduce_(a, name='a')
    print(a.asnumpy())

if __name__ == '__main__':
    main()