"""Read-only SQL over an analytics Postgres. Guardrails: SELECT-only,
statement timeout, row cap. The `sql` route uses this."""
import os
import re

import asyncpg

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|grant|truncate|copy)\b", re.I)


class SQLTool:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ.get(
            "ANALYTICS_DSN", "postgresql://postgres:dev@localhost:5432/analytics")

    async def query(self, sql: str, max_rows: int = 50) -> list[dict]:
        if FORBIDDEN.search(sql) or not sql.strip().lower().startswith("select"):
            raise ValueError("Only SELECT statements are permitted")
        conn = await asyncpg.connect(self.dsn, timeout=10,
                                     command_timeout=15)
        try:
            rows = await conn.fetch(sql)
            return [dict(r) for r in rows[:max_rows]]
        finally:
            await conn.close()
