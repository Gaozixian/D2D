import yaml

def load_config(config_path="./config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        # 解析 YAML 为字典
        config = yaml.safe_load(f)
    return config