import pandas as pd
from pathlib import Path
from typing import List
import re

import src.database.database_schema as db_schema
from src.database.connect_2_postgresDB import PostgresDB
from src.configs.config import Config


class EmbeddingUploader:

    def __init__(self, mini_batch_names: List[str]):
        self.config = Config(stage="EMBEDDING UPLOADER")
        self.base_dir = self.config.base_dir
        self.db_config = self.config.db_config
        self.scan_type_is_all = self.config.scan_type_is_all
        self.table_name = db_schema.get_table_name()
        self.table_column_names = db_schema.get_column_names()
        self.logger = self.config.logger
        self.embedded_data_path = None
        self.mini_batch_names = mini_batch_names
        # Create database connection
        self.db_connector = PostgresDB(
            dbname=self.db_config['dbname'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            host=self.db_config['host']
        )

    def validate_vector(self, value: str) -> bool:
        """Validate if a string is a valid vector format (e.g., '[0.1, 0.2, ...]')."""
        if not isinstance(value, str):
            return False
        # Check if string matches vector format: [number, number, ...]
        vector_pattern = r'^\[([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?,\s*)*[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?\]$'
        return bool(re.match(vector_pattern, value.strip()))

    def validate_csv_columns(self, csv_path: Path) -> bool:
        """Validate that CSV columns match the table's column names."""
        try:
            self.logger.info(f"Validating CSV columns for file: {csv_path}")
            with open(csv_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip().split('\t')
            if set(header) != set(self.table_column_names):
                self.logger.error(f"CSV columns {header} do not match table columns {self.table_column_names}")
                return False
            self.logger.info("CSV column validation passed")
            return True
        except Exception as e:
            self.logger.error(f"Error validating CSV columns for {csv_path}: {e}")
            return False

    def validate_csv_data(self, csv_path: Path) -> bool:
        """Validate CSV data to ensure vector columns contain valid vector strings."""
        try:
            self.logger.info(f"Validating CSV data for file: {csv_path}")
            df = pd.read_csv(csv_path, sep='\t', encoding='utf-8')
            vector_columns = ['embedded_design_images']
            text_columns = ['filename']  # Adjust if names differ

            # Validate vector columns
            for col in vector_columns:
                if col not in df.columns:
                    self.logger.error(f"Column '{col}' missing in CSV file")
                    return False
                invalid_rows = df[~df[col].apply(self.validate_vector)]
                if not invalid_rows.empty:
                    self.logger.error(f"Invalid vector values in column '{col}' at rows: {invalid_rows.index.tolist()}")
                    for idx in invalid_rows.index[:5]:  # Log up to 5 invalid values
                        self.logger.error(f"Row {idx}: {invalid_rows.loc[idx, col]}")
                    return False

            # Validate text columns (basic check for non-empty strings)
            for col in text_columns:
                if col not in df.columns:
                    continue  # Skip if column not present (optional columns)
                invalid_rows = df[df[col].str.strip() == '']
                if not invalid_rows.empty:
                    self.logger.warning(f"Empty values in column '{col}' at rows: {invalid_rows.index.tolist()}")

            self.logger.info("CSV data validation passed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error validating CSV data for {csv_path}: {e}")
            return False

    def create_table_if_needed(self):
        """Create table if it doesn't exist."""
        create_table_sql = db_schema.get_create_table_sql(if_not_exists=True)

        # Drop table if needed
        if self.scan_type_is_all:
            drop_table_sql = db_schema.get_drop_table_name()
            self.db_connector.cursor.execute(drop_table_sql)
            self.logger.success(f"Table '{self.table_name}' deleted successfully!")

            # Create table
            self.db_connector.cursor.execute(create_table_sql)
            self.db_connector.conn.commit()
            self.logger.success(f"Table '{self.table_name}' created successfully!")
        else:
            self.logger.info(f"Table '{self.table_name}' exists!")

    def upload_embeddings(self, mini_batch_name: str):
        """Upload embedding data from CSV to database using upsert to handle conflicts."""
        self.logger.info("Saving data to table...")
        self.embedded_data_path = Path(self.base_dir / f"data/information/{mini_batch_name}/2_embedded_data.csv")

        if not self.embedded_data_path.exists():
            self.logger.warning(f"CSV file not found at: {self.embedded_data_path}. Skipping uploading process.")
            return

        # Validate CSV columns and data before uploading
        if not self.validate_csv_columns(self.embedded_data_path):
            raise ValueError(f"CSV column validation failed for {self.embedded_data_path}")
        if not self.validate_csv_data(self.embedded_data_path):
            raise ValueError(f"CSV data validation failed for {self.embedded_data_path}")

        try:
            # Get target table columns and exclude created_at, updated_at
            target_columns = self.table_column_names
            excluded_columns = {'created_at', 'updated_at'}
            insert_columns = [col for col in target_columns if col not in excluded_columns]
            col_names = ', '.join(target_columns)
            unique_key = "id"

            # Create temporary table without created_at and updated_at
            self.logger.info("Creating temporary table...")
            create_temp_table_query = f"""
                CREATE TEMP TABLE tmp_table 
                    (LIKE {self.table_name} INCLUDING DEFAULTS EXCLUDING CONSTRAINTS)
                    ON COMMIT DROP;
            """
            self.db_connector.cursor.execute(create_temp_table_query)

            # Explicitly specify column order in COPY command
            copy_columns = ', '.join(target_columns)
            self.logger.info("Copying data into temporary table...")
            with open(self.embedded_data_path, 'r', encoding='utf-8') as csv_file:
                self.db_connector.cursor.copy_expert(
                    f"COPY tmp_table ({copy_columns}) FROM STDIN WITH (FORMAT csv, DELIMITER '\t', HEADER TRUE);",
                    csv_file
                )
            self.logger.success(f"Data copied successfully for {mini_batch_name}!")

            # Build the upsert query
            update_fields = [col for col in insert_columns if col != unique_key]
            set_clauses = [f"{col} = EXCLUDED.{col}" for col in update_fields]
            set_clauses.append(
                """updated_at = to_char((CURRENT_TIMESTAMP AT TIME ZONE 'UTC' + INTERVAL '7 hours'), 'YYYY-MM-DD"T"HH24:MI:SS.MSOF')"""
            )
            set_clause = ", ".join(set_clauses)

            upsert_query = f"""
                INSERT INTO {self.table_name} ({col_names})
                SELECT {col_names} FROM tmp_table
                ON CONFLICT ({unique_key})
                DO UPDATE SET {set_clause}
            """

            # Execute upsert
            self.logger.info("Performing upsert...")
            self.db_connector.cursor.execute(upsert_query)
            self.db_connector.conn.commit()
            self.logger.success(f"Data uploaded successfully for {mini_batch_name}!")

        except Exception as e:
            self.logger.error(f"An error occurred during upload: {e}")
            self.db_connector.conn.rollback()
            raise

    def verify_upload(self):
        """Verify the number of records uploaded."""
        self.db_connector.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name};")
        row_count = self.db_connector.cursor.fetchone()[0]
        self.logger.info(f"Number of records in the table: {row_count}")
        return row_count

    def run(self):
        """Main process to handle the embedding upload workflow."""
        try:
            self.db_connector.connect()

            # Check if delete current table or not
            self.create_table_if_needed()

            # Upload data to vectorDB - Sequences update
            for mini_batch_name in self.mini_batch_names:
                self.logger.info(f"Start uploading data to PG for {mini_batch_name}...")
                self.upload_embeddings(mini_batch_name)

            record_count = self.verify_upload()

            return {
                'status': 'success',
                'record_count': record_count,
                'message': 'Embedding upload completed successfully'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error during embedding upload: {str(e)}'
            }

        finally:
            if self.db_connector:
                self.db_connector.close()

def main():
    config = Config()
    result = EmbeddingUploader(config, []).run()
    print(result)

if __name__ == "__main__":
    main()