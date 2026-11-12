# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from rsl_rl.networks.normalization import EmpiricalNormalization


def test_reload_after_inference_mode_update() -> None:
    """Running statistics must remain writable when a trained checkpoint is loaded."""
    normalizer = EmpiricalNormalization(2)
    with torch.inference_mode():
        normalizer.update(torch.tensor([[1.0, 0.0], [3.0, 4.0]]))
    checkpoint = {key: value.clone() for key, value in normalizer.state_dict().items()}
    normalizer.load_state_dict(checkpoint)
    torch.testing.assert_close(normalizer.mean, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(normalizer.std, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(normalizer(torch.tensor([[2.0, 4.0]])), torch.tensor([[0.0, 2.0 / 2.01]]))
