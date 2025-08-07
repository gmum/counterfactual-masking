from pathlib import Path
import random
import yaml
import numpy as np
import torch
from loguru import logger


def load_config(config_file_path: Path | str = Path.cwd()/'config.yaml') -> dict:
    """
    Loads a YAML configuration file.

    Args:
        config_file_path (Path): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    if isinstance(config_file_path, str):
        config_file_path = Path(config_file_path)

    if not config_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found at '{config_file_path}'.")

    config = yaml.safe_load(config_file_path.read_text())

    return config


def get_nested(config: dict, *keys, default = None):
    """
    Safely retrieves a nested value from a dictionary using a sequence of keys.
    """
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            logger.warning(f"No value found for {'::'.join(keys)}, defaulting to {default}.")
            return default
    return config


def select_device(config: dict) -> torch.device:
    """
    Automatically selects the torch device based on config and availability.

    Priority:
        1. Config-specified device.
        2. CUDA if available.
        3. CPU fallback.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        torch.device: The selected device.
    """
    if (device_str := get_nested(config, 'MODEL', 'TRAINING', 'device')):
        try:
            device = torch.device(device_str)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid device name in config: '{device_str}'.") from exc
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}")

    return device


def set_random_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
