import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, *keys):
        value = self.config
        for key in keys:
            value = value[key]
        return value