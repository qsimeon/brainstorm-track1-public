"""
Configuration management for local environment variables.

This module loads environment-specific configuration from .env.local
without requiring external dependencies like python-dotenv.
"""

import os
from pathlib import Path


def load_env_file(env_file: Path = None) -> dict[str, str]:
    """
    Load environment variables from a .env.local file.

    Args:
        env_file: Path to .env.local file. If None, looks for .env.local
                 in the repository root.

    Returns:
        Dictionary of environment variables loaded from the file.
    """
    if env_file is None:
        repo_root = Path(__file__).parent.parent
        env_file = repo_root / ".env.local"

    env_vars = {}
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    env_vars[key] = value

    return env_vars


def get_checkpoint_dir() -> Path:
    """
    Get the checkpoint directory for saving best models.

    Reads from CHECKPOINT_DIR environment variable, with fallback to ./checkpoints.
    Creates the directory if it doesn't exist.

    Returns:
        Path to checkpoint directory.
    """
    # Load from .env.local
    env_vars = load_env_file()
    checkpoint_dir_str = env_vars.get("CHECKPOINT_DIR") or os.getenv("CHECKPOINT_DIR")

    # Fallback to ./checkpoints
    if not checkpoint_dir_str:
        checkpoint_dir_str = "./checkpoints"

    checkpoint_dir = Path(checkpoint_dir_str).expanduser().resolve()

    # Create directory if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir


def get_num_workers() -> int:
    """
    Get the number of workers for data loading.

    Reads from NUM_WORKERS environment variable, defaults to 0 (no multiprocessing).

    Returns:
        Number of workers.
    """
    env_vars = load_env_file()
    num_workers_str = env_vars.get("NUM_WORKERS") or os.getenv("NUM_WORKERS", "0")

    try:
        return int(num_workers_str)
    except ValueError:
        return 0


if __name__ == "__main__":
    # Test the configuration loading
    print("Checkpoint directory:", get_checkpoint_dir())
    print("Number of workers:", get_num_workers())
    print("All env vars:", load_env_file())
