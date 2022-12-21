from typing import Optional

from habitat_baselines.common.baseline_registry import BaselineRegistry
from habitat_eaif.il.objectnav.policy import ILPolicy


class CustomBaselineRegistry(BaselineRegistry):
    @classmethod
    def register_il_policy(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a IL policy with :p:`name`.
        :param name: Key with which the policy will be registered.
            If :py:`None` will use the name of the class
        .. code:: py
            from eai.objectnav_il.policy import ILPolicy
            from habitat_baselines.common.baseline_registry import (
                baseline_registry
            )
            @baseline_registry.register_il_policy
            class MyPolicy(ILPolicy):
                pass
            # or
            @baseline_registry.register_il_policy(name="MyPolicyName")
            class MyPolicy(Policy):
                pass
        """
        from habitat_eaif.il.objectnav.policy import ILPolicy

        return cls._register_impl("policy", to_register, name, assert_type=ILPolicy)


custom_baseline_registry = CustomBaselineRegistry()
