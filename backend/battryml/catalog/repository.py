from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from battryml.catalog.database import CatalogDatabase


def utc_now() -> datetime:
    return datetime.now(UTC)


class CatalogRepository:
    def __init__(self, database: CatalogDatabase):
        self.database = database

    def replace_run_bundle(
        self,
        run: dict[str, Any],
        artifacts: Sequence[dict[str, Any]],
        datasets: Sequence[dict[str, Any]],
    ) -> None:
        columns = tuple(run)
        placeholders = ", ".join("?" for _ in columns)
        with self.database.connection() as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute("DELETE FROM artifacts WHERE run_id = ?", [run["run_id"]])
                connection.execute("DELETE FROM datasets WHERE run_id = ?", [run["run_id"]])
                connection.execute("DELETE FROM analysis_runs WHERE run_id = ?", [run["run_id"]])
                connection.execute(
                    f"INSERT INTO analysis_runs ({', '.join(columns)}) VALUES ({placeholders})",
                    list(run.values()),
                )
                if artifacts:
                    artifact_columns = tuple(artifacts[0])
                    artifact_placeholders = ", ".join("?" for _ in artifact_columns)
                    connection.executemany(
                        f"INSERT INTO artifacts ({', '.join(artifact_columns)}) "
                        f"VALUES ({artifact_placeholders})",
                        [list(item.values()) for item in artifacts],
                    )
                if datasets:
                    dataset_columns = tuple(datasets[0])
                    dataset_placeholders = ", ".join("?" for _ in dataset_columns)
                    connection.executemany(
                        f"INSERT INTO datasets ({', '.join(dataset_columns)}) "
                        f"VALUES ({dataset_placeholders})",
                        [list(item.values()) for item in datasets],
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def refresh_batteries(self, source_root: str) -> None:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("DELETE FROM batteries")
            connection.execute(
                """
                INSERT INTO batteries (
                    serial, source_root, first_observed_at, last_observed_at,
                    latest_run_id, run_count, updated_at
                )
                SELECT
                    serial,
                    ?,
                    min(observed_start_at),
                    max(observed_end_at),
                    arg_max(run_id, run_created_at),
                    count(*),
                    ?
                FROM analysis_runs
                GROUP BY serial
                """,
                [source_root, now],
            )

    def list_batteries(self) -> list[dict[str, Any]]:
        return self.database.rows(
            """
            SELECT b.*, r.model_family AS latest_model_family,
                   r.validation_mae_soh_pct AS latest_validation_mae_soh_pct,
                   r.loss_slope_per_1000ah AS latest_loss_slope_per_1000ah
            FROM batteries b
            LEFT JOIN analysis_runs r ON r.run_id = b.latest_run_id
            ORDER BY b.serial
            """
        )

    def list_runs(
        self,
        *,
        serial: str | None = None,
        model_family: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if serial:
            clauses.append("serial = ?")
            parameters.append(serial)
        if model_family:
            clauses.append("model_family = ?")
            parameters.append(model_family)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(max(1, min(limit, 500)))
        return self.database.rows(
            f"""
            SELECT run_id, source_run_id, serial, pipeline_key, model_family, status,
                   nominal_capacity_ah, training_rows, validation_rows,
                   validation_mae_soh_pct, validation_max_error_soh_pct,
                   loss_slope_per_1000ah, forecast_rows, observed_start_at,
                   observed_end_at, run_created_at
            FROM analysis_runs
            {where}
            ORDER BY run_created_at DESC, run_id DESC
            LIMIT ?
            """,
            parameters,
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.database.row("SELECT * FROM analysis_runs WHERE run_id = ?", [run_id])

    def list_latest_runs(self, *, model_family: str) -> list[dict[str, Any]]:
        return self.database.rows(
            """
            SELECT run_id, source_run_id, serial, pipeline_key, model_family, status,
                   nominal_capacity_ah, training_rows, validation_rows,
                   validation_mae_soh_pct, validation_max_error_soh_pct,
                   loss_slope_per_1000ah, forecast_rows, observed_start_at,
                   observed_end_at, run_created_at
            FROM analysis_runs
            WHERE model_family = ?
            QUALIFY row_number() OVER (
                PARTITION BY serial ORDER BY run_created_at DESC, run_id DESC
            ) = 1
            ORDER BY serial
            """,
            [model_family],
        )

    def get_summary(self, run_id: str) -> dict[str, Any] | None:
        row = self.database.row(
            "SELECT summary_json FROM analysis_runs WHERE run_id = ?",
            [run_id],
        )
        return json.loads(row["summary_json"]) if row else None

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        return self.database.rows(
            """
            SELECT artifact_id, run_id, kind, name, path, media_type,
                   size_bytes, modified_at
            FROM artifacts
            WHERE run_id = ?
            ORDER BY kind, name
            """,
            [run_id],
        )

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        return self.database.row("SELECT * FROM artifacts WHERE artifact_id = ?", [artifact_id])

    def list_datasets(self, run_id: str) -> list[dict[str, Any]]:
        return self.database.rows(
            """
            SELECT dataset_id, run_id, serial, stage, name, path, format,
                   row_count, schema_json, observed_start_at, observed_end_at
            FROM datasets
            WHERE run_id = ?
            ORDER BY stage, name
            """,
            [run_id],
        )

    def find_artifact(self, run_id: str, kinds: Sequence[str]) -> dict[str, Any] | None:
        if not kinds:
            return None
        placeholders = ", ".join("?" for _ in kinds)
        return self.database.row(
            f"""
            SELECT * FROM artifacts
            WHERE run_id = ? AND kind IN ({placeholders})
            ORDER BY CASE kind {self._kind_order_sql(kinds)} ELSE 999 END
            LIMIT 1
            """,
            [run_id, *kinds],
        )

    @staticmethod
    def _kind_order_sql(kinds: Sequence[str]) -> str:
        return " ".join(f"WHEN '{kind}' THEN {index}" for index, kind in enumerate(kinds))

    def create_job(self, job: dict[str, Any]) -> None:
        columns = tuple(job)
        placeholders = ", ".join("?" for _ in columns)
        with self.database.connection() as connection:
            connection.execute(
                f"INSERT INTO pipeline_jobs ({', '.join(columns)}) VALUES ({placeholders})",
                list(job.values()),
            )

    def update_job(self, job_id: str, **changes: Any) -> None:
        if not changes:
            return
        assignments = ", ".join(f"{column} = ?" for column in changes)
        with self.database.connection() as connection:
            connection.execute(
                f"UPDATE pipeline_jobs SET {assignments} WHERE job_id = ?",
                [*changes.values(), job_id],
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.database.row("SELECT * FROM pipeline_jobs WHERE job_id = ?", [job_id])

    def list_jobs(
        self,
        limit: int = 100,
        *,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE archived_at IS NULL"
        return self.database.rows(
            f"SELECT * FROM pipeline_jobs {where} ORDER BY created_at DESC LIMIT ?",
            [max(1, min(limit, 500))],
        )

    def list_device_name_overrides(self) -> dict[str, str]:
        return {
            str(item["serial"]): str(item["display_name"])
            for item in self.database.rows(
                "SELECT serial, display_name FROM device_name_overrides"
            )
        }

    def set_device_name_override(self, serial: str, display_name: str) -> None:
        self.database.rows(
            """
            INSERT INTO device_name_overrides (serial, display_name, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT (serial) DO UPDATE SET
                display_name = excluded.display_name,
                updated_at = excluded.updated_at
            RETURNING serial
            """,
            [serial, display_name, utc_now()],
        )

    def clear_device_name_override(self, serial: str) -> None:
        with self.database.connection() as connection:
            connection.execute(
                "DELETE FROM device_name_overrides WHERE serial = ?",
                [serial],
            )

    def archive_job(self, job_id: str) -> dict[str, Any] | None:
        item = self.get_job(job_id)
        if item is None:
            return None
        if item["status"] in {"queued", "running"}:
            raise ValueError("Active jobs cannot be archived")
        self.update_job(job_id, archived_at=utc_now())
        return self.get_job(job_id)

    def delete_job(self, job_id: str) -> bool:
        item = self.get_job(job_id)
        if item is None:
            return False
        if item["status"] in {"queued", "running"}:
            raise ValueError("Active jobs cannot be removed")
        with self.database.connection() as connection:
            connection.execute("DELETE FROM pipeline_jobs WHERE job_id = ?", [job_id])
        return True

    def list_unfinished_jobs(self) -> list[dict[str, Any]]:
        return self.database.rows(
            """
            SELECT * FROM pipeline_jobs
            WHERE status IN ('queued', 'running')
            ORDER BY created_at
            """
        )

    def list_device_inventory(self) -> list[dict[str, Any]]:
        return self.database.rows("SELECT * FROM device_inventory ORDER BY serial")

    def get_device_inventory(self, serial: str) -> dict[str, Any] | None:
        return self.database.row(
            "SELECT * FROM device_inventory WHERE serial = ?",
            [serial],
        )

    def upsert_device_inventory(self, item: dict[str, Any]) -> None:
        columns = tuple(item)
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(
            f"{column} = excluded.{column}" for column in columns if column != "serial"
        )
        with self.database.connection() as connection:
            connection.execute(
                f"""
                INSERT INTO device_inventory ({', '.join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (serial) DO UPDATE SET {updates}
                """,
                list(item.values()),
            )

    def counts(self) -> dict[str, int]:
        row = self.database.row(
            """
            SELECT
                (SELECT count(*) FROM batteries) AS batteries,
                (SELECT count(*) FROM analysis_runs) AS runs,
                (SELECT count(*) FROM artifacts) AS artifacts,
                (SELECT count(*) FROM datasets) AS datasets,
                (SELECT count(*) FROM pipeline_jobs) AS jobs
            """
        )
        return {key: int(value) for key, value in (row or {}).items()}
