#!/usr/bin/env python3
# Author: Duo MA
# Email: maduo@cuhk.edu.cn

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from overrides import final
from overrides import override as override  # noqa: F401
from torch import device, dtype
from typing_extensions import TypeAlias

finaloverride = final

Device: TypeAlias = device

DataType: TypeAlias = dtype
