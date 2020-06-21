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

class DoubleMomTrainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer='sgd', optimizer_params=None, global_optimizer='nag', global_optimizer_params=None, local_sgd_interval=10):
        # optimizer, optimizer_params for local optimizer
        # global_optimizer, global_optimizer_params for global optimizer

        super(DoubleMomTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)

        # initialize global optimizer
        global_optimizer_params = global_optimizer_params if global_optimizer_params else {}
        self._init_global_optimizer(global_optimizer, global_optimizer_params)
        self._local_sgd_interval = local_sgd_interval
        self._local_sgd_counter = 0
        
        self._update_on_kvstore = False

        self._cache_initialized = False
        self._params_cache = []

        # need to double check self._scale
        # self._scale /= size()

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

        if not self._cache_initialized:
            self._init_cache()

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        # # cache the synchronous params
        # if self._local_sgd_counter == 0:
        #     for i, param in enumerate(self._params):
        #         if param.grad_req != 'null':
        #             self._params_cache[i][:] = param.list_data()[0]
        #             self._grads_cache[i][:] = 0

        # assume lr does not change during local steps

        # self._allreduce_grads()
        self._update(ignore_stale_grad)

        # local sgd
        self._local_sgd_counter += 1
        if self._local_sgd_counter == self._local_sgd_interval:
            self.global_step(1. / self.learning_rate)
    
    def global_step(self, rescale_grad=1.0):
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

        self._global_optimizer.rescale_grad = rescale_grad

        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                param.list_grad()[0][:] = self._params_cache[i] - param.list_data()[0]
                allreduce_(param.list_grad()[0], average=True,
                           name=param.name, priority=-i)
        self._global_update()

        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                # copy the globally updated params back to local
                param.list_data()[0][:] = self._params_cache[i]


        self._local_sgd_counter = 0

    def _init_cache(self):
        if self._params_cache == []:
            # initialize the cached params
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    self._params_cache.append(param.list_data()[0].copy())
                else:
                    self._params_cache.append([])
        self._cache_initialized = True


    def _init_global_optimizer(self, optimizer, optimizer_params):
        param_dict = {i: param for i, param in enumerate(self._params)}
        if isinstance(optimizer, opt.Optimizer):
            assert not optimizer_params, \
                "optimizer_params must be None if optimizer is an instance of " \
                "Optimizer instead of str"
            self._global_optimizer = optimizer
            # param_dict must not be deep copied, so that if user mutate the lr_mult
            # or wd_mult of some parameters, it takes effect.
            self._global_optimizer.param_dict = param_dict
        else:
            self._global_optimizer = opt.create(optimizer, param_dict=param_dict,
                                         **optimizer_params)
        self._global_updaters = [opt.get_updater(self._global_optimizer) \
                            for _ in self._contexts]

    def _global_update(self):
        global_updates = [[] for _ in self._global_updaters]

        # set wd
        self._global_optimizer.wd = 0

        for i, param in enumerate(self._params):
            if param.grad_req == 'null':
                continue

            # need to double check stale grad
            # if not ignore_stale_grad:
            #     for data in param._check_and_get(param._data, list):
            #         if not data._fresh_grad:
            #             raise UserWarning(
            #                 "Gradient of Parameter `%s` on context %s has not been updated "
            #                 "by backward since last `step`. This could mean a bug in your "
            #                 "model that made it only use a subset of the Parameters (Blocks) "
            #                 "for this iteration. If you are intentionally only using a subset, "
            #                 "call step with ignore_stale_grad=True to suppress this "
            #                 "warning and skip updating of Parameters with stale gradient" \
            #                 %(param.name, str(data.context)))

            # if self._kvstore and self._update_on_kvstore:
            #     continue

            # for upd, arr, grad in zip(updates, param.list_data(), param.list_grad()):
            #     if not ignore_stale_grad or arr._fresh_grad:
            #         upd.append((i, grad, arr))
            #         arr._fresh_grad = False

            # only for hvd
            global_updates[0].append((i, param.list_grad()[0], self._params_cache[i]))
            # self._params_cache[i]._fresh_grad = False

        if not (self._kvstore and self._update_on_kvstore):
            for updater, upd in zip(self._global_updaters, global_updates):
                if upd:
                    i, g, w = zip(*upd)
                    updater(i, g, w)

    def reset_momentum(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                # mp later
                self._updaters[0].states[i][:] = 0

    @property
    def global_learning_rate(self):
        if not isinstance(self._global_optimizer, opt.Optimizer):
            raise UserWarning("Global optimizer has to be defined before its learning "
                              "rate can be accessed.")

        return self._global_optimizer.learning_rate

    def set_global_learning_rate(self, lr):
        """Sets a new learning rate of the optimizer.

        Parameters
        ----------
        lr : float
            The new learning rate of the optimizer.
        """
        if not isinstance(self._global_optimizer, opt.Optimizer):
            raise UserWarning("Global optimizer has to be defined before its learning "
                              "rate is mutated.")
        self._global_optimizer.set_learning_rate(lr)


