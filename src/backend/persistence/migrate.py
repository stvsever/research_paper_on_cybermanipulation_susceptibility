from __future__ import annotations

import argparse
from pathlib import Path

from src.backend.persistence.db import connect


SQL_DIR = Path(__file__).resolve().parent / "sql"


def migration_files() -> list[Path]:
    return sorted(SQL_DIR.glob("*.sql"))


def _sql_statements(path: Path) -> list[str]:
    return [statement.strip() for statement in path.read_text(encoding="utf-8").split(";") if statement.strip()]


def migrate_up(database_url: str | None = None) -> list[str]:
    applied: list[str] = []
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            for path in migration_files():
                version = path.stem
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
                )
                cur.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (version,))
                if cur.fetchone():
                    continue
                for statement in _sql_statements(path):
                    cur.execute(statement)
                cur.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (version,))
                applied.append(version)
        conn.commit()
    return applied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Postgres migrations for pipeline artifact persistence.")
    parser.add_argument("command", choices=["up"])
    parser.add_argument("--database-url", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "up":
        applied = migrate_up(args.database_url)
        print(f"Applied migrations: {', '.join(applied) if applied else 'none'}")


if __name__ == "__main__":
    main()
