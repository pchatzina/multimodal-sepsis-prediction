# src/utils/database.py
import os
from sqlalchemy import create_engine, text
from src.utils.config import Config


def get_engine():
    """Factory to create the SQLAlchemy engine based on Config."""
    return create_engine(Config.get_db_url())
