from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import duckdb


class StorageError(RuntimeError):
    """Base exception for AVF-TRPDE persistent storage errors."""


class RecordExistsError(StorageError):
    """Raised when a persistent record already exists."""


class RecordNotFoundError(StorageError):
    """Raised when a persistent record does not exist."""


class DuckDBStorage:
    """
    Central persistent storage layer for AVF-TRPDE.

    The storage layer owns persistence only. It does not contain
    forecasting, risk, portfolio, or research logic.
    """

    _write_lock = RLock()

    def __init__(
        self,
        database_path: str | Path | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]

        if database_path is None:
            database_path = (
                project_root
                / "storage"
                / "duckdb"
                / "avf_trpde.duckdb"
            )

        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialize()

    # ------------------------------------------------------------------
    # Connection / initialization
    # ------------------------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(
            str(self.database_path),
        )

    def initialize(self) -> None:
        """Create persistent tables when they do not already exist."""

        with self._write_lock:
            connection = self._connect()

            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datasets (
                        name VARCHAR PRIMARY KEY,
                        source VARCHAR NOT NULL,
                        frequency VARCHAR NOT NULL,
                        assets VARCHAR NOT NULL,
                        start_timestamp VARCHAR NOT NULL,
                        end_timestamp VARCHAR NOT NULL,
                        row_count BIGINT NOT NULL,
                        fingerprint VARCHAR NOT NULL,
                        status VARCHAR NOT NULL,
                        created_at VARCHAR NOT NULL
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS experiments (
                        name VARCHAR PRIMARY KEY,
                        description VARCHAR NOT NULL,
                        models VARCHAR NOT NULL,
                        metrics VARCHAR NOT NULL,
                        risk_measures VARCHAR NOT NULL,
                        parameters VARCHAR NOT NULL,
                        metadata VARCHAR NOT NULL,
                        status VARCHAR NOT NULL,
                        created_at VARCHAR NOT NULL
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS forecasts (
                        experiment_name VARCHAR NOT NULL,
                        dataset_name VARCHAR NOT NULL,
                        model VARCHAR NOT NULL,
                        forecasts VARCHAR NOT NULL,
                        observation_count BIGINT NOT NULL,
                        asset_count BIGINT NOT NULL,
                        status VARCHAR NOT NULL,
                        created_at VARCHAR NOT NULL,
                        PRIMARY KEY (experiment_name, model)
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS risk_results (
                        experiment_name VARCHAR NOT NULL,
                        dataset_name VARCHAR NOT NULL,
                        model VARCHAR NOT NULL,
                        risk_points VARCHAR NOT NULL,
                        observation_count BIGINT NOT NULL,
                        asset_count BIGINT NOT NULL,
                        status VARCHAR NOT NULL,
                        created_at VARCHAR NOT NULL,
                        PRIMARY KEY (experiment_name, model)
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS portfolio_results (
                        experiment_name VARCHAR NOT NULL,
                        dataset_name VARCHAR NOT NULL,
                        strategy VARCHAR NOT NULL,
                        positions VARCHAR NOT NULL,
                        observation_count BIGINT NOT NULL,
                        asset_count BIGINT NOT NULL,
                        status VARCHAR NOT NULL,
                        created_at VARCHAR NOT NULL,
                        PRIMARY KEY (experiment_name, strategy)
                    )
                    """
                )

                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reports (
                        experiment_name VARCHAR NOT NULL,
                        dataset_name VARCHAR NOT NULL,
                        report_type VARCHAR NOT NULL,
                        generated_at VARCHAR NOT NULL,
                        sections VARCHAR NOT NULL,
                        status VARCHAR NOT NULL,
                        created_at VARCHAR NOT NULL,
                        PRIMARY KEY (
                            experiment_name,
                            report_type
                        )
                    )
                    """
                )

            finally:
                connection.close()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json_dump(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _json_load(value: str) -> Any:
        return json.loads(value)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Dataset persistence
    # ------------------------------------------------------------------

    def save_dataset(
        self,
        record: dict[str, Any],
    ) -> None:
        with self._write_lock:
            connection = self._connect()

            try:
                connection.execute(
                    """
                    INSERT INTO datasets (
                        name,
                        source,
                        frequency,
                        assets,
                        start_timestamp,
                        end_timestamp,
                        row_count,
                        fingerprint,
                        status,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record["name"],
                        record["source"],
                        record["frequency"],
                        self._json_dump(record["assets"]),
                        str(record["start_timestamp"]),
                        str(record["end_timestamp"]),
                        record["row_count"],
                        record["fingerprint"],
                        record.get("status", "registered"),
                        self._timestamp(),
                    ],
                )
            except duckdb.ConstraintException as exc:
                raise RecordExistsError(
                    f"Dataset '{record['name']}' already exists."
                ) from exc
            finally:
                connection.close()

    def get_dataset(
        self,
        name: str,
    ) -> dict[str, Any] | None:
        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT
                    name,
                    source,
                    frequency,
                    assets,
                    start_timestamp,
                    end_timestamp,
                    row_count,
                    fingerprint,
                    status
                FROM datasets
                WHERE name = ?
                """,
                [name],
            ).fetchone()

            if row is None:
                return None

            return {
                "name": row[0],
                "source": row[1],
                "frequency": row[2],
                "assets": self._json_load(row[3]),
                "start_timestamp": row[4],
                "end_timestamp": row[5],
                "row_count": row[6],
                "fingerprint": row[7],
                "status": row[8],
            }
        finally:
            connection.close()

    def list_datasets(self) -> list[dict[str, Any]]:
        connection = self._connect()

        try:
            rows = connection.execute(
                """
                SELECT
                    name,
                    source,
                    frequency,
                    assets,
                    start_timestamp,
                    end_timestamp,
                    row_count,
                    fingerprint,
                    status
                FROM datasets
                ORDER BY created_at
                """
            ).fetchall()

            return [
                {
                    "name": row[0],
                    "source": row[1],
                    "frequency": row[2],
                    "assets": self._json_load(row[3]),
                    "start_timestamp": row[4],
                    "end_timestamp": row[5],
                    "row_count": row[6],
                    "fingerprint": row[7],
                    "status": row[8],
                }
                for row in rows
            ]
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # Experiment persistence
    # ------------------------------------------------------------------

    def save_experiment(
        self,
        record: dict[str, Any],
    ) -> None:
        with self._write_lock:
            connection = self._connect()

            try:
                connection.execute(
                    """
                    INSERT INTO experiments (
                        name,
                        description,
                        models,
                        metrics,
                        risk_measures,
                        parameters,
                        metadata,
                        status,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record["name"],
                        record.get("description", ""),
                        self._json_dump(record["models"]),
                        self._json_dump(record["metrics"]),
                        self._json_dump(
                            record.get("risk_measures", [])
                        ),
                        self._json_dump(
                            record.get("parameters", {})
                        ),
                        self._json_dump(
                            record.get("metadata", {})
                        ),
                        record.get("status", "configured"),
                        self._timestamp(),
                    ],
                )
            except duckdb.ConstraintException as exc:
                raise RecordExistsError(
                    f"Experiment '{record['name']}' already exists."
                ) from exc
            finally:
                connection.close()

    def get_experiment(
        self,
        name: str,
    ) -> dict[str, Any] | None:
        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT
                    name,
                    description,
                    models,
                    metrics,
                    risk_measures,
                    parameters,
                    metadata,
                    status
                FROM experiments
                WHERE name = ?
                """,
                [name],
            ).fetchone()

            if row is None:
                return None

            return {
                "name": row[0],
                "description": row[1],
                "models": self._json_load(row[2]),
                "metrics": self._json_load(row[3]),
                "risk_measures": self._json_load(row[4]),
                "parameters": self._json_load(row[5]),
                "metadata": self._json_load(row[6]),
                "status": row[7],
            }
        finally:
            connection.close()

    def list_experiments(self) -> list[dict[str, Any]]:
        connection = self._connect()

        try:
            rows = connection.execute(
                """
                SELECT
                    name,
                    description,
                    models,
                    metrics,
                    risk_measures,
                    parameters,
                    metadata,
                    status
                FROM experiments
                ORDER BY created_at
                """
            ).fetchall()

            return [
                {
                    "name": row[0],
                    "description": row[1],
                    "models": self._json_load(row[2]),
                    "metrics": self._json_load(row[3]),
                    "risk_measures": self._json_load(row[4]),
                    "parameters": self._json_load(row[5]),
                    "metadata": self._json_load(row[6]),
                    "status": row[7],
                }
                for row in rows
            ]
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # Forecast persistence
    # ------------------------------------------------------------------

    def save_forecast(
        self,
        record: dict[str, Any],
    ) -> None:
        with self._write_lock:
            connection = self._connect()

            try:
                connection.execute(
                    """
                    INSERT INTO forecasts (
                        experiment_name,
                        dataset_name,
                        model,
                        forecasts,
                        observation_count,
                        asset_count,
                        status,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record["experiment_name"],
                        record["dataset_name"],
                        record["model"],
                        self._json_dump(record["forecasts"]),
                        record["observation_count"],
                        record["asset_count"],
                        record["status"],
                        self._timestamp(),
                    ],
                )
            except duckdb.ConstraintException as exc:
                raise RecordExistsError(
                    "Forecast for experiment "
                    f"'{record['experiment_name']}' and model "
                    f"'{record['model']}' already exists."
                ) from exc
            finally:
                connection.close()

    def get_forecast(
        self,
        experiment_name: str,
        model: str,
    ) -> dict[str, Any] | None:
        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT
                    experiment_name,
                    dataset_name,
                    model,
                    forecasts,
                    observation_count,
                    asset_count,
                    status
                FROM forecasts
                WHERE experiment_name = ?
                  AND model = ?
                """,
                [experiment_name, model],
            ).fetchone()

            if row is None:
                return None

            return {
                "experiment_name": row[0],
                "dataset_name": row[1],
                "model": row[2],
                "forecasts": self._json_load(row[3]),
                "observation_count": row[4],
                "asset_count": row[5],
                "status": row[6],
            }
        finally:
            connection.close()

    def list_forecasts(
        self,
        experiment_name: str | None = None,
    ) -> list[dict[str, Any]]:
        connection = self._connect()

        try:
            if experiment_name is None:
                rows = connection.execute(
                    """
                    SELECT
                        experiment_name,
                        dataset_name,
                        model,
                        forecasts,
                        observation_count,
                        asset_count,
                        status
                    FROM forecasts
                    ORDER BY created_at
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT
                        experiment_name,
                        dataset_name,
                        model,
                        forecasts,
                        observation_count,
                        asset_count,
                        status
                    FROM forecasts
                    WHERE experiment_name = ?
                    ORDER BY created_at
                    """,
                    [experiment_name],
                ).fetchall()

            return [
                {
                    "experiment_name": row[0],
                    "dataset_name": row[1],
                    "model": row[2],
                    "forecasts": self._json_load(row[3]),
                    "observation_count": row[4],
                    "asset_count": row[5],
                    "status": row[6],
                }
                for row in rows
            ]
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # Risk persistence
    # ------------------------------------------------------------------

    def save_risk(
        self,
        record: dict[str, Any],
    ) -> None:
        with self._write_lock:
            connection = self._connect()

            try:
                connection.execute(
                    """
                    INSERT INTO risk_results (
                        experiment_name,
                        dataset_name,
                        model,
                        risk_points,
                        observation_count,
                        asset_count,
                        status,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record["experiment_name"],
                        record["dataset_name"],
                        record["model"],
                        self._json_dump(record["risk_points"]),
                        record["observation_count"],
                        record["asset_count"],
                        record["status"],
                        self._timestamp(),
                    ],
                )
            except duckdb.ConstraintException as exc:
                raise RecordExistsError(
                    "Risk result for experiment "
                    f"'{record['experiment_name']}' and model "
                    f"'{record['model']}' already exists."
                ) from exc
            finally:
                connection.close()

    def get_risk(
        self,
        experiment_name: str,
        model: str,
    ) -> dict[str, Any] | None:
        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT
                    experiment_name,
                    dataset_name,
                    model,
                    risk_points,
                    observation_count,
                    asset_count,
                    status
                FROM risk_results
                WHERE experiment_name = ?
                  AND model = ?
                """,
                [experiment_name, model],
            ).fetchone()

            if row is None:
                return None

            return {
                "experiment_name": row[0],
                "dataset_name": row[1],
                "model": row[2],
                "risk_points": self._json_load(row[3]),
                "observation_count": row[4],
                "asset_count": row[5],
                "status": row[6],
            }
        finally:
            connection.close()

    def list_risks(
        self,
        experiment_name: str | None = None,
    ) -> list[dict[str, Any]]:
        connection = self._connect()

        try:
            if experiment_name is None:
                rows = connection.execute(
                    """
                    SELECT
                        experiment_name,
                        dataset_name,
                        model,
                        risk_points,
                        observation_count,
                        asset_count,
                        status
                    FROM risk_results
                    ORDER BY created_at
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT
                        experiment_name,
                        dataset_name,
                        model,
                        risk_points,
                        observation_count,
                        asset_count,
                        status
                    FROM risk_results
                    WHERE experiment_name = ?
                    ORDER BY created_at
                    """,
                    [experiment_name],
                ).fetchall()

            return [
                {
                    "experiment_name": row[0],
                    "dataset_name": row[1],
                    "model": row[2],
                    "risk_points": self._json_load(row[3]),
                    "observation_count": row[4],
                    "asset_count": row[5],
                    "status": row[6],
                }
                for row in rows
            ]
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # Portfolio persistence
    # ------------------------------------------------------------------

    def save_portfolio(
        self,
        record: dict[str, Any],
    ) -> None:
        with self._write_lock:
            connection = self._connect()

            try:
                connection.execute(
                    """
                    INSERT INTO portfolio_results (
                        experiment_name,
                        dataset_name,
                        strategy,
                        positions,
                        observation_count,
                        asset_count,
                        status,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record["experiment_name"],
                        record["dataset_name"],
                        record["strategy"],
                        self._json_dump(record["positions"]),
                        record["observation_count"],
                        record["asset_count"],
                        record["status"],
                        self._timestamp(),
                    ],
                )
            except duckdb.ConstraintException as exc:
                raise RecordExistsError(
                    "Portfolio result for experiment "
                    f"'{record['experiment_name']}' and strategy "
                    f"'{record['strategy']}' already exists."
                ) from exc
            finally:
                connection.close()

    def get_portfolio(
        self,
        experiment_name: str,
        strategy: str,
    ) -> dict[str, Any] | None:
        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT
                    experiment_name,
                    dataset_name,
                    strategy,
                    positions,
                    observation_count,
                    asset_count,
                    status
                FROM portfolio_results
                WHERE experiment_name = ?
                  AND strategy = ?
                """,
                [experiment_name, strategy],
            ).fetchone()

            if row is None:
                return None

            return {
                "experiment_name": row[0],
                "dataset_name": row[1],
                "strategy": row[2],
                "positions": self._json_load(row[3]),
                "observation_count": row[4],
                "asset_count": row[5],
                "status": row[6],
            }
        finally:
            connection.close()

    def list_portfolios(
        self,
        experiment_name: str | None = None,
    ) -> list[dict[str, Any]]:
        connection = self._connect()

        try:
            if experiment_name is None:
                rows = connection.execute(
                    """
                    SELECT
                        experiment_name,
                        dataset_name,
                        strategy,
                        positions,
                        observation_count,
                        asset_count,
                        status
                    FROM portfolio_results
                    ORDER BY created_at
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT
                        experiment_name,
                        dataset_name,
                        strategy,
                        positions,
                        observation_count,
                        asset_count,
                        status
                    FROM portfolio_results
                    WHERE experiment_name = ?
                    ORDER BY created_at
                    """,
                    [experiment_name],
                ).fetchall()

            return [
                {
                    "experiment_name": row[0],
                    "dataset_name": row[1],
                    "strategy": row[2],
                    "positions": self._json_load(row[3]),
                    "observation_count": row[4],
                    "asset_count": row[5],
                    "status": row[6],
                }
                for row in rows
            ]
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    def save_report(
        self,
        record: dict[str, Any],
    ) -> None:
        with self._write_lock:
            connection = self._connect()

            try:
                connection.execute(
                    """
                    INSERT INTO reports (
                        experiment_name,
                        dataset_name,
                        report_type,
                        generated_at,
                        sections,
                        status,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record["experiment_name"],
                        record["dataset_name"],
                        record["report_type"],
                        str(record["generated_at"]),
                        self._json_dump(record["sections"]),
                        record["status"],
                        self._timestamp(),
                    ],
                )
            except duckdb.ConstraintException as exc:
                raise RecordExistsError(
                    "Report for experiment "
                    f"'{record['experiment_name']}' and type "
                    f"'{record['report_type']}' already exists."
                ) from exc
            finally:
                connection.close()

    def get_report(
        self,
        experiment_name: str,
        report_type: str,
    ) -> dict[str, Any] | None:
        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT
                    experiment_name,
                    dataset_name,
                    report_type,
                    generated_at,
                    sections,
                    status
                FROM reports
                WHERE experiment_name = ?
                  AND report_type = ?
                """,
                [experiment_name, report_type],
            ).fetchone()

            if row is None:
                return None

            return {
                "experiment_name": row[0],
                "dataset_name": row[1],
                "report_type": row[2],
                "generated_at": row[3],
                "sections": self._json_load(row[4]),
                "status": row[5],
            }
        finally:
            connection.close()

    def list_reports(
        self,
        experiment_name: str | None = None,
    ) -> list[dict[str, Any]]:
        connection = self._connect()

        try:
            if experiment_name is None:
                rows = connection.execute(
                    """
                    SELECT
                        experiment_name,
                        dataset_name,
                        report_type,
                        generated_at,
                        sections,
                        status
                    FROM reports
                    ORDER BY created_at
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT
                        experiment_name,
                        dataset_name,
                        report_type,
                        generated_at,
                        sections,
                        status
                    FROM reports
                    WHERE experiment_name = ?
                    ORDER BY created_at
                    """,
                    [experiment_name],
                ).fetchall()

            return [
                {
                    "experiment_name": row[0],
                    "dataset_name": row[1],
                    "report_type": row[2],
                    "generated_at": row[3],
                    "sections": self._json_load(row[4]),
                    "status": row[5],
                }
                for row in rows
            ]
        finally:
            connection.close()


storage = DuckDBStorage()