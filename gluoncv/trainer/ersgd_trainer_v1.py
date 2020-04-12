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

class ERSGDTrainerV1(mx.gluon.Trainer):
    def __init__(self, params, optimizer='ERSGDV1', optimizer_params=None, input_sparse_ratio=1, output_sparse_ratio=1, layer_sparse_ratio=1):

        super(ERSGDTrainerV1, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)
        
        self._update_on_kvstore = False

        # ER-SGD
        self._input_sparse_ratio = input_sparse_ratio
        self._output_sparse_ratio = output_sparse_ratio
        self._layer_sparse_ratio = layer_sparse_ratio
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

        self._allreduce_grads()

    def _allreduce_grads(self):
        
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                if param.list_grad()[0].stype == 'default':
                    # ER-SGD
                    r, _, _ = self._updaters[0].states[i]

                    if random.uniform(0,1) <= self._layer_sparse_ratio:
                        # compress
                        input_size = r.shape[0]
                        k1 = max(1, round(input_size*self._input_sparse_ratio))
                        sparse_input_begin = random.choice(range(math.ceil(input_size/k1))) * k1
                        sparse_input_end = min(sparse_input_begin + k1, input_size)
                        if len(r.shape) > 1:
                            output_size = r.shape[1]
                            k2 = max(1, round(output_size*self._output_sparse_ratio))
                            sparse_output_begin = random.choice(range(math.ceil(output_size/k2))) * k2
                            sparse_output_end = min(sparse_output_begin + k2, output_size)

                            r_sync = r[sparse_input_begin:sparse_input_end,sparse_output_begin:sparse_output_end]
                            param.list_data()[0][sparse_input_begin:sparse_input_end,sparse_output_begin:sparse_output_end] += r_sync
                            # partial sync
                            allreduce_(r_sync, average=True,
                                    name=str(i), priority=-i)
                            
                            param.list_data()[0][sparse_input_begin:sparse_input_end,sparse_output_begin:sparse_output_end] -= r_sync
                            r[sparse_input_begin:sparse_input_end,sparse_output_begin:sparse_output_end] = 0
                        else:

                            r_sync = r[sparse_input_begin:sparse_input_end]
                            param.list_data()[0][sparse_input_begin:sparse_input_end] += r_sync
                            # partial sync
                            allreduce_(r_sync, average=True,
                                    name=str(i), priority=-i)
                            
                            param.list_data()[0][sparse_input_begin:sparse_input_end] -= r_sync
                            r[sparse_input_begin:sparse_input_end] = 0

                        # communication counter
                        self._comm_counter += r_sync.size * 2
                else:
                    raise ValueError("Cannot pull row_sparse parameters for local SGD")
    
    def _init_params_cache(self):
        if self._params_cache == []:
            # initialize the states
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


