# src/utils/database.py
import pandas as pd
from sqlalchemy import create_engine, text
from src.utils.config import Config


def get_engine():
    """Factory to create the SQLAlchemy engine based on Config."""
    return create_engine(Config.get_db_url())


def query_to_df(query_str: str) -> pd.DataFrame:
    """
    Executes a SQL query and returns the results as a pandas DataFrame.
    """
    engine = get_engine()
    with engine.connect() as conn:
        # read_sql automatically handles column headers and type conversion
        df = pd.read_sql(text(query_str), conn)
    return df
