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
from mxnet.model import _create_kvstore, _create_sparse_kvstore
from mxnet.gluon.parameter import ParameterDict, Parameter

import mxnet as mx
import types
import warnings
import math

import horovod.mxnet as hvd
from horovod.mxnet.mpi_ops import allreduce_

class Distributed2StepsTrainer(mx.gluon.Trainer):
    # only works with LocalAdaAlter
    def __init__(self, params, optimizer, pre_optimizer=None, optimizer_params=None, sync_grad = True, reset_interval=0, save_prev_lr=False, sparse_ratio=0, sparse_lower=True):

        self._pre_optimizer = pre_optimizer

        # ersgd
        self._sync_grad = sync_grad
        self._reset_interval = reset_interval
        self._reset_counter = 0
        self._sparse_ratio = sparse_ratio
        self._sparse_lower = sparse_lower

        super(Distributed2StepsTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None, update_on_kvstore = False)
        
        self._update_on_kvstore = False

        # efsgd
        self._save_prev_lr = save_prev_lr

        self._hvd_param_buf = {}


        # self._scale /= hvd.size()

        # print(self._local_sgd_interval)

    def _init_optimizer(self, optimizer, optimizer_params):
        param_dict = {i: param for i, param in enumerate(self._params)}
        if isinstance(optimizer, opt.Optimizer):
            assert not optimizer_params, \
                "optimizer_params must be None if optimizer is an instance of " \
                "Optimizer instead of str"
            self._optimizer = optimizer
            # param_dict must not be deep copied, so that if user mutate the lr_mult
            # or wd_mult of some parameters, it takes effect.
            self._optimizer.param_dict = param_dict
        else:
            self._optimizer = opt.create(optimizer, param_dict=param_dict,
                                         **optimizer_params)

        self._updaters = [opt.get_updater(self._optimizer) \
                            for _ in self._contexts]

        # for efsgd and signum
        if self._pre_optimizer is not None:
            # # debug
            # print('found a second optimizer:', self._pre_optimizer)
            # pre_optimizer_name = self._pre_optimizer
            self._pre_optimizer = opt.create(self._pre_optimizer, param_dict=param_dict,
                                         **optimizer_params)
            self._pre_updaters = [opt.get_updater(self._pre_optimizer) \
                            for _ in self._contexts]

            # if str.lower(pre_optimizer_name).startswith('ersgd') or str.lower(pre_optimizer_name).startswith('spsgd'):
            self._optimizer.pre_updater = self._pre_updaters[0]
            # sparse compression
            self._pre_optimizer.sparse_index = []
            if self._sparse_ratio > 0:
                # debug
                print('use sparsity in ERSGD')
                param_idx_list = []
                for i, param in enumerate(self._params):
                    if param.grad_req != 'null':
                        param_idx_list.append(i)
                if self._sparse_ratio >= 1:
                    self._pre_optimizer.sparse_index = param_idx_list
                else:
                    if self._sparse_lower:
                        sparse_index_threshold = param_idx_list[round(len(param_idx_list)*self._sparse_ratio)]
                        for i, param in enumerate(self._params):
                            if param.grad_req != 'null':
                                if i <= sparse_index_threshold:
                                    self._pre_optimizer.sparse_index.append(i)
                    else:
                        sparse_index_threshold = param_idx_list[round(len(param_idx_list)*(1-self._sparse_ratio))]
                        for i, param in enumerate(self._params):
                            if param.grad_req != 'null':
                                if i >= sparse_index_threshold:
                                    self._pre_optimizer.sparse_index.append(i)
            self._optimizer.sparse_index = self._pre_optimizer.sparse_index
        else:
            self._pre_optimizer = None
            self._pre_updaters = None


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
        # if self._local_sgd_interval == 0:
        #     rescale_grad = self._scale / batch_size / hvd.size()
        # else:
        #     rescale_grad = self._scale / batch_size
        rescale_grad = self._scale / batch_size
        self._pre_optimizer.rescale_grad = rescale_grad
        # self._check_and_rescale_grad(rescale_grad)

        # sync lr between pre-optimizer and post-optimizer
        if self._pre_optimizer is not None:
            self._pre_optimizer.set_learning_rate(self._optimizer.learning_rate)

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        # before allreduce the gradients
        if self._pre_optimizer is not None:
            updates = [[] for _ in self._pre_updaters]

            for i, param in enumerate(self._params):
                if param.grad_req == 'null':
                    continue

                for upd, arr, grad in zip(updates, param.list_data(), param.list_grad()):
                    if not ignore_stale_grad or arr._fresh_grad:
                        upd.append((i, grad, arr))
                        # arr._fresh_grad = True

            if not (self._kvstore and self._update_on_kvstore):
                for updater, upd in zip(self._pre_updaters, updates):
                    if upd:
                        i, w, g = zip(*upd)
                        updater(i, w, g)
    
        if self._sync_grad:
            # print('_allreduce_grads')
            self._allreduce_grads()

        self._update(ignore_stale_grad)

        # efsgd
        if self._save_prev_lr:
            self._optimizer.prev_lr = self._optimizer.learning_rate
            if self._pre_optimizer is not None:
                self._pre_optimizer.prev_lr = self._pre_optimizer.learning_rate

        if self._reset_interval > 0:
            # local sgd
            self._reset_counter += 1
            if self._reset_counter == self._reset_interval:
                # print('local sgd')
                self._reset_counter = 0
                # synchronization
                self.allreduce_params()
                self.allreduce_states()
                # indicate that the parameters are synchronized in the current iteration
                return True
            return False
        return True

    def _allreduce_grads(self):
        # sort needed for Python < 3.6 is not guaranteed
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                if self._pre_optimizer is not None and i in self._pre_optimizer.sparse_index:
                    # sparsity for ersgd
                    continue
                allreduce_(param.list_grad()[0], average=True,
                           name=str(i), priority=-i)

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
                if self._pre_optimizer is not None and (i in self._pre_optimizer.sparse_index or self._optimizer.compress):
                    hvd.allreduce_(param.list_data()[0], average=True, 
                                            name=str(len(self._params) + i), priority=-i)
                    self._optimizer.bit_counter += (param.list_data()[0].size) * 32 * 2
                # for j in range(1, len(param.list_data())):
                #     param.list_data()[0].copyto(param.list_data()[j])

    def allreduce_states(self):
        for i, param in reversed(list(enumerate(self._params))):
            if param.grad_req != 'null':
                if self._pre_optimizer is not None and (i in self._pre_optimizer.sparse_index or self._optimizer.compress):
                    state_array = self._updaters[0].states[i]
                    idx = i+len(self._params)*2
                    if param._stype == 'default':
                        hvd.allreduce_(state_array, average=True, 
                                    name=str(idx), priority=i-len(self._params)*2)
                        self._optimizer.bit_counter += (state_array.size) * 32 * 2
                    else:
                        raise ValueError("Cannot pull row_sparse parameters for local SGD")

    # def init_states(self):
    #     # self._hvd_param_buf = {}
    #     mx.nd.waitall()
    #     self._hvd_param_buf.clear()
    #     mx.nd.waitall()
    #     for i, param in reversed(list(enumerate(self._params))):
    #         if param.grad_req != 'null':
    #             self._updaters[0].states[i] = (self._updaters[0].states[i][0], self._updaters[0].states[i][0].copy())
    #     mx.nd.waitall()