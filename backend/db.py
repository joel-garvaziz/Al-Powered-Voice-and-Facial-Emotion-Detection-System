import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_db():
    """Create and return a MySQL database connection."""
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "Root12345"),
        database=os.getenv("MYSQL_DATABASE", "emosense"),
    )


def close_db(conn):
    """Safely close a database connection."""
    if conn and conn.is_connected():
        conn.close()
