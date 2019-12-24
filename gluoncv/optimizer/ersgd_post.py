# coding: utf-8
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

"""AdaAlter optimizer"""

from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray
from mxnet.ndarray import square, power, sqrt, maximum, minimum, clip, sign, norm
from mxnet.ndarray import sparse

import horovod.mxnet as hvd

__all__ = ['ERSGDPost']


@register
class ERSGDPost(Optimizer):
    """AdaAlter optimizer.
    TODO(xcong): update the description
    This class implements the AdaGrad optimizer described in *Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization*, and available at
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
    This optimizer updates each weight by::
        grad = clip(grad * rescale_grad, clip_gradient)
        div = grad / sqrt(history + float_stable_eps)
        weight += (div + weight * wd) * -lr
        history += square(grad)
    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.
    See Also
    ----------
    :meth:`mxnet.ndarray.sparse.adagrad_update`.
    Parameters
    ----------
    eps: float, optional
        Initial value of the history accumulator. Avoids division by 0.
    """
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=False, compress=True, **kwargs):
        super(ERSGDPost, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self.compressed_grad = dict()
        self.compress = compress
        self.bit_counter = 0.
        # debug
        if self.nesterov:
            print('use Nesterov momentum')

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=weight.stype)
        return momentum

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)

        mom = state
        error = self.pre_updater.states[index]

        # compress
        if index not in self.compressed_grad:
            self.compressed_grad[index] = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=weight.stype)
        compressed_grad = self.compressed_grad[index]
        if index in self.sparse_index:
            # grad[:] = 0
            self.bit_counter += 0
        else:
            if self.compress:
                sign(grad, out=compressed_grad)
                compressed_grad[:] *= (norm(grad, ord=1) / grad.size)
                # state is the error
                if index % hvd.size() != hvd.rank():
                    grad[:] = compressed_grad
                self.bit_counter += (state.size + 32) * 2
            else:
                # error[:] = 0
                self.bit_counter += (state.size) * 32 * 2
        grad[:] += error

        mom[:] *= self.momentum
        mom[:] += grad

        # update
        if self.nesterov:
            weight[:] -= lr * (grad + mom * self.momentum)
        else:
            weight[:] -= lr * mom

