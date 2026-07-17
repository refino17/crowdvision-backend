"""
CrowdVision AI security configuration helpers.

This module intentionally avoids external dependencies such as python-dotenv.
It loads simple KEY=VALUE lines from .env when present, while still allowing
real environment variables to override local file values.
"""

import os
from pathlib import Path

_ENV_LOADED = False


def project_root():
    return Path(__file__).resolve().parent.parent


def load_env_file(path=None):
    global _ENV_LOADED

    if _ENV_LOADED:
        return

    env_path = Path(path) if path else project_root() / ".env"

    if not env_path.exists():
        _ENV_LOADED = True
        return

    try:
        with env_path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as error:
        print(f"Unable to load .env file: {error}")

    _ENV_LOADED = True


def get_env(name, default=""):
    load_env_file()
    return os.environ.get(name, default)


def get_env_bool(name, default=False):
    value = str(get_env(name, "")).strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on", "enabled"}


def get_edge_api_key():
    return str(get_env("CROWDVISION_EDGE_API_KEY", "")).strip()


def edge_key_enabled():
    return bool(get_edge_api_key())


def mask_secret(value, visible=4):
    text = str(value or "")
    if not text:
        return "Not configured"
    if len(text) <= visible * 2:
        return "•" * len(text)
    return f"{text[:visible]}{'•' * 8}{text[-visible:]}"