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

class LocalSGDTrainerV1(mx.gluon.Trainer):
    def __init__(self, params, optimizer='nag', optimizer_params=None, local_sgd_interval=4):

        super(LocalSGDTrainerV1, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)
        
        self._update_on_kvstore = False

        # local-SGD
        self._local_sgd_interval = local_sgd_interval
        self._local_sgd_counter = 0

        self._params_cache_to_init = True
        self._params_cache = []

        # communication counter
        self._comm_counter = 0.

    def step(self, batch_size, ignore_stale_grad=False):
        """Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.
        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        rescale_grad = self._scale / batch_size
        self._check_and_rescale_grad(rescale_grad)

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()
        
        if self._params_cache_to_init:
            self._init_params_cache()

        self._update(ignore_stale_grad)

        # local sgd
        self._local_sgd_counter += 1
        if self._local_sgd_counter == self._local_sgd_interval:
            self._local_sgd_counter = 0
            self._allreduce_params()

    def _allreduce_params(self):
        
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                hvd.allreduce_(param.list_data()[0], average=True, 
                                name=str(i), priority=-i)

                # communication counter
                self._comm_counter += param.list_data()[0].size * 2

    def allreduce_params(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                hvd.allreduce_(param.list_data()[0], average=True, 
                                       name=str(i), priority=-i)

    def _init_states(self):
        if self._e == [] and self._x == []:
            # initialize the remaining error
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    self._e.append(zeros_like(param.list_data()[0]))
                    self._x.append(zeros_like(param.list_data()[0]))
                else:
                    self._e.append([])
                    self._x.append([])
        self._states_to_init = False
    
    def _init_params_cache(self):
        if self._params_cache == []:
            # initialize the cached params
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    self._params_cache.append(zeros_like(param.list_data()[0]))
                else:
                    self._params_cache.append([])
        self._params_cache_to_init = False
    
    def pre_test(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                self._params_cache[i][:] = param.list_data()[0]
                hvd.allreduce_(param.list_data()[0], average=True, 
                                       name=str(i), priority=-i)
    def post_test(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                param.list_data()[0][:] = self._params_cache[i]


