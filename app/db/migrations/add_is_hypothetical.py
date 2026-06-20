"""
Migration: add is_hypothetical column to hands table.
Run once: python -m app.db.migrations.add_is_hypothetical
"""
from app.db.session import engine
from sqlalchemy import text

def upgrade():
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE hands
                ADD COLUMN IF NOT EXISTS is_hypothetical BOOLEAN NOT NULL DEFAULT FALSE;
        """))
        conn.execute(text("""
            ALTER TABLE hands
                ALTER COLUMN session_id DROP NOT NULL;
            """))
        conn.commit()
        print("Migration complete: is_hypothetical added, session_id now nullable.")

    if __name__ == "__main__":
        upgrade()