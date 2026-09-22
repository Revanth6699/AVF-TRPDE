from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    app_name: str = "AVF-TRPDE"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False

    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    intraday_data_dir: Path = PROJECT_ROOT / "data" / "intraday"
    realized_data_dir: Path = PROJECT_ROOT / "data" / "realized"
    samples_data_dir: Path = PROJECT_ROOT / "data" / "samples"

    experiments_dir: Path = PROJECT_ROOT / "experiments"
    reports_dir: Path = PROJECT_ROOT / "reports"
    storage_dir: Path = PROJECT_ROOT / "storage"

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @classmethod
    def from_environment(cls) -> "Settings":
        """Create application settings from environment variables."""

        return cls(
            app_name=os.getenv(
                "AVF_APP_NAME",
                "AVF-TRPDE",
            ),
            app_version=os.getenv(
                "AVF_APP_VERSION",
                "1.0.0",
            ),
            environment=os.getenv(
                "AVF_ENVIRONMENT",
                "development",
            ),
            debug=_get_bool(
                "AVF_DEBUG",
                default=False,
            ),
            api_host=os.getenv(
                "AVF_API_HOST",
                "127.0.0.1",
            ),
            api_port=_get_int(
                "AVF_API_PORT",
                default=8000,
            ),
        )

    def ensure_directories(self) -> None:
        """Create runtime directories required by the application."""

        directories = (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.intraday_data_dir,
            self.realized_data_dir,
            self.samples_data_dir,
            self.experiments_dir,
            self.reports_dir,
            self.storage_dir,
        )

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


def _get_bool(
    name: str,
    *,
    default: bool,
) -> bool:
    """Read a boolean environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be a boolean value "
        f"(true/false, 1/0, yes/no, on/off)."
    )


def _get_int(
    name: str,
    *,
    default: int,
) -> int:
    """Read an integer environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer."
        ) from exc


settings = Settings.from_environment()