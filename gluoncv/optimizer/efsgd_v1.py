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
# from mxnet.ndarray import efsgd_pre_update
from mxnet.ndarray import efsgd_pre_update, mp_efsgd_pre_update

__all__ = ['EFSGDV1']

@register
class EFSGDV1(Optimizer):
    """The EF-SGD optimizer.
    """
    def __init__(self, momentum=0.9, nesterov=True,
                 **kwargs):
        super(EFSGDV1, self).__init__(**kwargs)
        self.momentum = momentum
        self.nesterov = nesterov

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == numpy.float16:
            weight_master_copy = weight.astype(numpy.float32)
            return (zeros(weight.shape, weight.context, dtype=weight.dtype), # e, remaining error
                    zeros(weight.shape, weight.context, dtype=numpy.float32), # m, momentum
                    zeros(weight.shape, weight.context, dtype=numpy.float32), # m_wd, momentum of weight decay
                    weight_master_copy) # the float32 copy of weight
        if weight.dtype == numpy.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "NAG optimizer")
        return self.create_state(index, weight)

    def create_state(self, _, weight):
        """state creation function."""
        return (zeros(weight.shape, weight.context, dtype=weight.dtype), # e, remaining error
                zeros(weight.shape, weight.context, dtype=weight.dtype), # m, momentum
                zeros(weight.shape, weight.context, dtype=weight.dtype)) # m_wd, momentum of weight decay

    def _update_impl(self, index, weight, grad, state, multi_precision=False):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        kwargs = {'momentum': self.momentum, 'nesterov': self.nesterov, 
                  'rescale_grad': self.rescale_grad}
        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if not multi_precision:
            e, m, m_wd = state
            efsgd_pre_update(weight, grad, e, m, m_wd, out=weight,
                        lr=lr, wd=wd, **kwargs)
        else:
            e, m, m_wd, w_32 = state
            mp_efsgd_pre_update(weight, grad, e, m, m_wd, w_32, out=weight,
                        lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        use_multi_precision = self.multi_precision and weight.dtype == numpy.float16 \
                                and isinstance(state, (tuple, list))
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

        

        
