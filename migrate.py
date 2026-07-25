import asyncio
from pathlib import Path

import asyncpg

from .config import get_settings

async def migrate() -> None:
    settings = get_settings()
    sql_path = Path(__file__).resolve().parents[2] / "supabase" / "migrations" / "001_youtube_rag.sql"
    if not sql_path.exists():
        raise RuntimeError(f"Migration file not found: {sql_path}")
    connection = await asyncpg.connect(settings.supabase_db_url, statement_cache_size=0)
    try:
        await connection.execute(sql_path.read_text(encoding="utf-8"))
    finally:
        await connection.close()

def main() -> None:
    asyncio.run(migrate())

if __name__ == "__main__":
    main()