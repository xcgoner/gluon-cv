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

# pylint: disable=wildcard-import
"""CV trainer."""

from .sgd_trainer import SGDTrainer

from .efsgd_trainer_v1 import EFSGDTrainerV1
from .ersgd_trainer_v1 import ERSGDTrainerV1
from .ersgd2_trainer_v1 import ERSGD2TrainerV1

from .ersgd_trainer_v2 import ERSGDTrainerV2
from .ersgd2_trainer_v2 import ERSGD2TrainerV2

from .qsparse_local_sgd_trainer_v1 import QSparseLocalSGDTrainerV1
from .partial_local_sgd_trainer_v1 import PartialLocalSGDTrainerV1
from .local_sgd_trainer_v1 import LocalSGDTrainerV1