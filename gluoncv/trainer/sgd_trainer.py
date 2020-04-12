# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=line-too-long
"""Parameter optimizer."""

from mxnet import optimizer as opt
from mxnet.gluon.parameter import ParameterDict, Parameter
from mxnet.ndarray import square, zeros_like, random_uniform
from mxnet.ndarray.contrib import boolean_mask

import mxnet as mx
import types
import warnings
import math
import random
import logging

import horovod.mxnet as hvd
from horovod.mxnet.mpi_ops import allreduce, allreduce_
from horovod.mxnet.mpi_ops import size

class SGDTrainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer='nag', optimizer_params=None):

        super(SGDTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)
        
        self._update_on_kvstore = False

        # self._scale /= size()

        # communication counter
        self._comm_counter = 0.

    def _allreduce_grads(self):

        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                allreduce_(param.list_grad()[0], average=True,
                           name=param.name, priority=-i)

                # communication counter
                self._comm_counter += param.list_grad()[0].size * 2


