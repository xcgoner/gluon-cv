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

"""Weight updating functions."""
import os
import warnings
import numpy
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray, full
from mxnet.ndarray import ersgd_pre_update

__all__ = ['ERSGDV1']

@register
class ERSGDV1(Optimizer):
    """The EF-SGD optimizer.
    """
    def __init__(self, learning_rate=0.1, momentum=0.9, nesterov=True,
                 version=0, **kwargs):
        super(ERSGDV1, self).__init__(learning_rate=learning_rate, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self.version = version

    def create_state(self, _, weight):
        """state creation function."""
        return (zeros(weight.shape, weight.context, dtype=weight.dtype), #r, remaining error
                zeros(weight.shape, weight.context, dtype=weight.dtype), #m, momentum
                zeros(weight.shape, weight.context, dtype=weight.dtype)) #m_wd, momentum of weight decay

    def update(self, index, weight, grad, state):
        """update function"""
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        kwargs = {'momentum': self.momentum, 'nesterov': self.nesterov, 
                  'version': self.version, 'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        r, m, m_wd = state
        ersgd_pre_update(weight, grad, r, m, m_wd, out=weight,
                    lr=lr, wd=wd, **kwargs)
