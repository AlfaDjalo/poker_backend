from sqlalchemy import create_engine, text

# Adjust port if you moved it to 5433
engine = create_engine("postgresql+psycopg2://poker:poker@localhost:5432/poker")

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")