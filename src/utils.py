"""Utility functions for interacting with the SQLite database."""
import io
import logging
import sqlite3
from typing import Any, List, Optional

import torch


def select_tensors(
        db_path: str,
        table_name: str,
        keys: List[str] = ['layer', 'pooling_method', 'tensor_dim', 'tensor'],
        sql_where: Optional[str] = None,
        ) -> List[Any]:
    """Select and return all tensors from the specified SQLite database and table.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to query.
        keys (List[str]): List of keys to select from the database.
        sql_where (str): Optional SQL WHERE clause to filter results.

    Returns:
        List[Any]: A list of tensors retrieved from the database.
    """
    if 'tensor' not in keys:
        logging.warning("'tensor' key should be included to retrieve tensors; automatically adding it.")
        keys.append('tensor')
    final_results = []
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        query = f'SELECT {", ".join(keys)} FROM {table_name}'
        if sql_where:
            assert sql_where.strip().lower().startswith('where'), "sql_where should start with 'WHERE'"
            query += f' {sql_where}'
        cursor.execute(query)
        results = cursor.fetchall()
        for row in results:
            result_item = {key: value for key, value in zip(keys, row)}
            result_item['tensor'] = torch.load(io.BytesIO(result_item['tensor']), map_location='cpu')
            final_results.append(result_item)
    return final_results
