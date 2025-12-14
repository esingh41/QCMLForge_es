# This file is part of tad-mctc.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Coordination number: Counting functions
=======================================

This module contains all the counting functions used throughout the projects of
the Grimme group. This includes the following counting functions:
- exponential (DFT-D3, EEQ)
- error function (DFT-D4)
- double exponential (GFN2-xTB)

Additionally, the analytical derivatives for the counting functions are also
provided and can be used for checking the autograd results.
"""
from __future__ import annotations

from math import pi, sqrt

import torch
from tad_dftd3 import defaults
from tad_dftd3.typing import (
    Tensor,
)
__all__ = [
    "exp_count",
    "dexp_count",
    "erf_count",
    "derf_count",
    "gfn2_count",
    "dgfn2_count",
]


def exp_count(
    r: Tensor, r0: Tensor, kcn: Tensor | float | int = defaults.KCN_D3
) -> Tensor:
    """
    Exponential counting function for coordination number contributions.

    Parameters
    ----------
    r : Tensor
        Internuclear distances.
    r0 : Tensor
        Covalent atomic radii (R_AB = R_A + R_B).
    kcn : Tensor | float | int, optional
        Steepness of the counting function. Defaults to
        :data:`tad_mctc.ncoord.defaults.KCN_D3`.

    Returns
    -------
    Tensor
        Count of coordination number contribution.
    """
    return 1.0 / (1.0 + torch.exp(-kcn * (torch.divide(r0, r) - 1.0)))



def dexp_count(
    r: Tensor, r0: Tensor, kcn: Tensor | float | int = defaults.KCN_D3
) -> Tensor:
    """
    Derivative of the exponential counting function w.r.t. the distance.

    Parameters
    ----------
    r : Tensor
        Internuclear distances.
    r0 : Tensor
        Covalent atomic radii (R_AB = R_A + R_B).
    kcn : Tensor | float | int, optional
        Steepness of the counting function. Defaults to
        :data:`tad_mctc.ncoord.defaults.KCN_D3`.

    Returns
    -------
    Tensor
        Derivative of count of coordination number contribution.
    """
    expterm = torch.exp(-kcn * (torch.divide(r0, r) - 1.0))
    return (-kcn * r0 * expterm) / (r**2 * ((expterm + 1.0) ** 2))


