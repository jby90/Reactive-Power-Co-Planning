import numpy as np
import torch

from vmod_actor import CapacityConditionedActor


def test_policy_parameter_iterator_excludes_critic_parameters():
    actor = CapacityConditionedActor(
        8, 3, [0.225, 0.0, 0.0], [1.5, 1.5, 1.0]
    )
    policy_ids = {id(parameter) for parameter in actor.policy_parameters()}
    expected_ids = {
        id(parameter)
        for name, parameter in actor.named_parameters()
        if name.startswith(("pi_fc1.", "pi_fc2.", "pi_mu.", "pi_log_std"))
    }

    assert policy_ids == expected_ids


def test_actor_applies_configured_observation_transform():
    actor = CapacityConditionedActor(4, 2, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    actor.configure_observation_transform(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([2.0, 2.0, 4.0, 4.0]),
        np.asarray([1.0, 0.0, 1.0, 0.0]),
    )
    transformed = actor._features(
        torch.tensor([3.0, 20.0, 7.0, 40.0]),
        torch.tensor([0.5, 0.5, 0.5]),
    )

    np.testing.assert_allclose(
        transformed.detach().numpy(),
        np.asarray([1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.5]),
    )
