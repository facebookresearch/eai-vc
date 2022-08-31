from typing import Dict, Optional, Type


class EnvRegistry:
    def __init__(self):
        self._envs: Dict[str, Type] = {}

    def register_env(self, register_name: str):
        def wrap(to_register_cls: Type):
            self._envs[register_name] = to_register_cls
            return to_register_cls

        return wrap

    def search_env(self, name: str) -> Optional[Type]:
        return self._envs.get(name, None)


full_env_registry = EnvRegistry()
