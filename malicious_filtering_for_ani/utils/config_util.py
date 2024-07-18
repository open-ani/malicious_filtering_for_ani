import yaml

"""
这个文件主要是用来读取配置文件的
"""


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
