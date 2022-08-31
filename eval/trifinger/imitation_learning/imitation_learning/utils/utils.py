from typing import Dict, Any


def compress_and_filter_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    ret_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            ret_d.update(compress_and_filter_dict(v))
        else:
            ret_d[k] = v
    return ret_d


class flatten_info_dict_reader:
    def __init__(self, keys=None):
        if keys is None:
            keys = []
        self.keys = keys

    def __call__(self, info_dict: dict, tensordict: "_TensorDict") -> "_TensorDict":
        if not self.keys:
            return tensordict

        if type(info_dict) is list:
            info_dict = info_dict[0]
        info_dict = compress_and_filter_dict(info_dict)
        for key in self.keys:
            if key in info_dict:
                tensordict[key] = info_dict[key]
        return tensordict
