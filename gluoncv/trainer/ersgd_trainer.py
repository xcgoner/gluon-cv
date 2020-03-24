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

class ERSGDTrainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, lr, optimizer_params=None, sparse_ratio=0, momentum=0.9, wd=0.0001, nesterov=False):

        super(ERSGDTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None, update_on_kvstore = False)
        
        self._update_on_kvstore = False

        # ER-SGD
        self._sparse_ratio = sparse_ratio
        self._momentum = momentum
        self._nesterov = nesterov
        self._lr = lr
        self._wd = wd
        self._states_to_init = True
        self._states = []

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

        if self._states_to_init:
            self._init_states()
    
        self._allreduce_grads()

    def _init_states(self):
        if self._states == []:
            # initialize the states
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    # \hat{x} and momentum
                    self._states.append([param.list_data()[0].copy(), zeros_like(param.list_grad()[0])])
                else:
                    self._states.append([])
        self._states_to_init = False

    def _allreduce_grads(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                if param.list_grad()[0].stype == 'default':
                    # ER-SGD
                    x_hat, m = self._states[i]
                    param.list_grad()[0][:] += self._wd * param.list_data()[0]
                    param.list_grad()[0][:] *= self._lr
                    m[:] *= self._momentum
                    m[:] += param.list_grad()[0]
                    # if self._nesterov:
                    #     param.list_grad()[0][:] = m * self._momentum + param.list_grad()[0] + self._lr * self._wd * param.list_data()[0] + x_hat - param.list_data()[0]
                    # else:
                    #     param.list_grad()[0][:] = m + self._lr * self._wd * param.list_data()[0] + x_hat - param.list_data()[0]

                    if self._nesterov:
                        param.list_grad()[0][:] = m * self._momentum + param.list_grad()[0] 
                    else:
                        param.list_grad()[0][:] = m
                    allreduce_(param.list_grad()[0], average=True,
                               name=str(i), priority=-i)
                    param.list_data()[0][:] -= param.list_grad()[0]

                    # # compress
                    # length = m.shape[0]
                    # g = param.list_grad()[0]
                    # k = round(length*self._sparse_ratio)
                    # sparse_mask = random.sample(range(length), k=k)

                    # # debug
                    # if k < 1:
                    #     logging.info('sparse ratio is too small')
                    
                    # # # debug
                    # # logging.info(random.sample(range(10), 4))
                    # # mx.nd.waitall()

                    # g_sync = g[sparse_mask]
                    # r = g.copy()
                    # r[sparse_mask] = 0
                    # # partial sync
                    # allreduce_(g_sync, average=True,
                    #            name=str(i), priority=-i)
                    # g[sparse_mask] = g_sync
                    # param.list_data()[0][:] = x_hat - g
                    # x_hat[:] = param.list_data()[0] + r
                else:
                    raise ValueError("Cannot pull row_sparse parameters for local SGD")

    def allreduce_params(self):
        """For each parameter, reduce the parameters from different contexts.
        Should be called after `autograd.backward()`, outside of `record()` scope,
        and before `trainer.update()`.
        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.
        """
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                hvd.allreduce_(param.list_data()[0], average=True, 
                                       name=str(i), priority=-i)
