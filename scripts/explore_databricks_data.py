#!/usr/bin/env python3
"""
Data exploration script for ICC Chatbot Databricks tables
This script connects to Databricks and explores the structure and content
of the geneva_convention_chunks and icc_judgments_chunks tables.
"""

import os
import pandas as pd
import databricks.sql as sql
from databricks.sdk import WorkspaceClient
import json

def connect_to_databricks():
    """Connect to Databricks using default CLI profile"""
    try:
        # Initialize workspace client with default profile
        w = WorkspaceClient(profile="DEFAULT")
        
        # Get connection details for SQL warehouse
        warehouse_id = "554667dc3febf474"
        server_hostname = "dbc-0619d7f5-0bda.cloud.databricks.com"
        
        # Get access token
        token = w.config.token
        
        return server_hostname, warehouse_id, token
    except Exception as e:
        print(f"Error connecting to Databricks: {e}")
        return None, None, None

def explore_table_structure(connection, table_name):
    """Explore the structure and sample data of a table"""
    print(f"\n{'='*60}")
    print(f"EXPLORING TABLE: {table_name}")
    print(f"{'='*60}")
    
    try:
        # Get table schema
        schema_query = f"DESCRIBE TABLE {table_name}"
        schema_df = pd.read_sql(schema_query, connection)
        print(f"\nSCHEMA for {table_name}:")
        print(schema_df.to_string(index=False))
        
        # Get row count
        count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        count_df = pd.read_sql(count_query, connection)
        print(f"\nTotal rows: {count_df['row_count'].iloc[0]}")
        
        # Get sample data
        sample_query = f"SELECT * FROM {table_name} LIMIT 5"
        sample_df = pd.read_sql(sample_query, connection)
        print(f"\nSample data (first 5 rows):")
        print(sample_df.to_string(index=False))
        
        # Get unique values for key columns (if they exist)
        try:
            # Check for common metadata columns
            metadata_columns = ['source_file', 'document_type', 'section', 'chunk_index', 'page_number']
            for col in metadata_columns:
                if col in sample_df.columns:
                    unique_query = f"SELECT DISTINCT {col}, COUNT(*) as count FROM {table_name} GROUP BY {col} ORDER BY count DESC LIMIT 10"
                    unique_df = pd.read_sql(unique_query, connection)
                    print(f"\nUnique values in {col}:")
                    print(unique_df.to_string(index=False))
        except Exception as e:
            print(f"Could not analyze metadata columns: {e}")
        
        # Analyze text content length
        try:
            length_query = f"""
            SELECT 
                MIN(LENGTH(text)) as min_text_length,
                MAX(LENGTH(text)) as max_text_length,
                AVG(LENGTH(text)) as avg_text_length,
                MIN(LENGTH(summary)) as min_summary_length,
                MAX(LENGTH(summary)) as max_summary_length,
                AVG(LENGTH(summary)) as avg_summary_length
            FROM {table_name}
            """
            length_df = pd.read_sql(length_query, connection)
            print(f"\nText content analysis:")
            print(length_df.to_string(index=False))
        except Exception as e:
            print(f"Could not analyze text lengths: {e}")
            
    except Exception as e:
        print(f"Error exploring table {table_name}: {e}")

def main():
    """Main function to explore both tables"""
    print("ICC Chatbot Data Exploration")
    print("Connecting to Databricks...")
    
    server_hostname, warehouse_id, token = connect_to_databricks()
    
    if not all([server_hostname, warehouse_id, token]):
        print("Failed to connect to Databricks. Please check your configuration.")
        return
    
    # Connect to SQL warehouse
    try:
        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            access_token=token
        )
        print("Successfully connected to Databricks SQL warehouse!")
        
        # Explore both tables
        tables = [
            "icc_chatbot.search_model.geneva_convention_chunks",
            "icc_chatbot.search_model.icc_judgments_chunks"
        ]
        
        for table in tables:
            explore_table_structure(connection, table)
        
        connection.close()
        print("\nData exploration completed!")
        
    except Exception as e:
        print(f"Error connecting to SQL warehouse: {e}")

if __name__ == "__main__":
    main()
