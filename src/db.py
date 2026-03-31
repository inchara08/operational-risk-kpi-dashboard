"""Shared PostgreSQL connection utility."""
from __future__ import annotations
import os
import psycopg2


def get_connection(cfg: dict):
    """
    Connect to PostgreSQL. On macOS with Homebrew (no PGPASSWORD set),
    omitting host forces a Unix domain socket connection (peer auth — no password needed).
    """
    db = cfg["database"]
    password = os.environ.get("PGPASSWORD", "")
    host = db["host"]
    if host in ("localhost", "127.0.0.1") and not password:
        return psycopg2.connect(dbname=db["dbname"], user=db["user"])
    return psycopg2.connect(
        host=host, port=db["port"],
        dbname=db["dbname"], user=db["user"],
        password=password,
    )
