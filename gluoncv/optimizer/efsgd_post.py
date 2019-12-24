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

__all__ = ['EFSGDPost']


@register
class EFSGDPost(Optimizer):
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
    def __init__(self, learning_rate=0.01, momentum=0.9, compress=True, **kwargs):
        super(EFSGDPost, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum
        self.prev_lr = learning_rate
        self.compress = compress

        self.bit_counter = 0

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
        wd = self._get_wd(index)

        grad[:] += state * (self.prev_lr / lr)
        # compress
        if index in self.sparse_index:
            state[:] = grad
            grad = 0
        else: 
            if self.compress:
                state[:] = grad
                sign(grad, out=grad)
                grad[:] *= (norm(state, ord=1) / state.size)
                # state is the error
                state[:] -= grad
                self.bit_counter += (state.size + 32) * 2
            else:
                state[:] = 0
                self.bit_counter += (state.size) * 32 * 2

        # update
        weight[:] -= lr * grad

