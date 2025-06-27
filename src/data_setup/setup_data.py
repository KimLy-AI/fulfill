from src.configs.config import Config
import src.database.database_schema as db_schema
from src.database.connect_2_postgresDB import PostgresDB
from src.utils.utils import (
    FileUtils,
    PipelineUtils
)
import pandas as pd
import numpy as np
import json
from pathlib import Path
import shutil


class SetupData:

    def __init__(self):
        self.config = Config(stage="SETUP")
        self.db_config = self.config.db_config
        self.db_connector = PostgresDB(
            dbname=self.db_config['dbname'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            host=self.db_config['host']
        )
        self.scan_type_is_all = self.config.scan_type_is_all
        self.processed_designs_query = db_schema.get_processed_designs()
        self.logger = self.config.logger
        self._setup_path()

    def _setup_path(self):
        # 0. Load config
        self.setup_data_config = PipelineUtils.load_config(config_path=self.config.setup_data_config_path)

        # gcs source raw data
        self.gcs_source_folder = f"{self.config.env}/{self.setup_data_config['gcs_raw_prefix']}/{self.config.batch_name}"
        # tmp location to load data
        self.local_tmp_path = self.config.base_dir / self.setup_data_config['local_tmp']
        self.local_tmp_download_path = self.local_tmp_path / "downloaded"
        self.local_tmp_extract_path = self.local_tmp_path / "extracted"
        self.local_tmp_information_path = self.local_tmp_download_path / "information.json"
        self.local_tmp_image_zip_path = self.local_tmp_download_path / "images.zip"
        self.local_tmp_images_path = self.local_tmp_extract_path / "images"
        # data location in mini-batch for training
        self.local_data_path = self.config.base_dir / self.setup_data_config['local_data']
        self.local_data_images_path = self.local_data_path / "images/design_flattened"
        self.local_information_path = self.local_data_path / "clean_metadata"
    @staticmethod
    def load_desgin_info(design_info_path):

        file_path = Path(design_info_path)
        with file_path.open("r", encoding="utf-8") as file:
            json_data = json.load(file)

        return pd.json_normalize(json_data, sep="_")

    def filter_new_design_info(self, design_info_path):
        """Returns only designs that haven't been processed before"""
        designs_info_df = self.load_designs_info(design_info_path)
        processed_designs_df = self.get_processed_designs()
        new_designs_info_condition = (
            ~designs_info_df["fileid"].isin(list(processed_designs_df["fileid"]))
        )
        return designs_info_df[new_designs_info_condition]

    @staticmethod
    def split_images_and_info_into_batches(batch_name: str, source_folder: str, image_dest_folder: str,
                                           info_dest_folder: str, df: pd.DataFrame, batch_size: int, logger):
        """
        Splits valid images and their corresponding information from a DataFrame into batches.
        Each batch is stored in a subfolder named 'batch1', 'batch2', etc.
        A CSV file for each batch contains design information.

        :param source_folder: Path to the folder containing images
        :param image_dest_folder: Path to the destination folder for image batches
        :param info_dest_folder: Path to the destination folder for information batches
        :param df: DataFrame containing design information
        :param batch_size: Number of records per batch
        """
        source_path = Path(source_folder)
        image_dest_path = Path(image_dest_folder)
        info_dest_path = Path(info_dest_folder)

        # Remove existing batch folders if they exist
        if image_dest_path.exists():
            shutil.rmtree(image_dest_path)
        if info_dest_path.exists():
            shutil.rmtree(info_dest_path)

        # Ensure destination directories exist
        image_dest_path.mkdir(parents=True, exist_ok=True)
        info_dest_path.mkdir(parents=True, exist_ok=True)

        # image_path_columns = ["design_images"]
        # if batch_name in ["batch2", "batch3"]:  # old logic
        #     # Process image paths
        #     for col in image_path_columns:
        #         df[f"{col}_name"] = df[col].astype(str).apply(
        #             lambda x: f"{Path(x).parent.name}_{Path(x).name}"
        #         )
        # else:
        #     def clean_web_name(web_url):
        #         """Extract filename from web_url without .html suffix."""
        #         return Path(web_url).stem if pd.notnull(web_url) else None
        #
        #     for col in image_path_columns:
        #         df[f"{col}_name"] = df.apply(
        #             lambda row: (
        #                 f"{clean_web_name(row['web_url'])}_{Path(row[col]).name}"
        #                 if pd.notnull(row[col]) and pd.notnull(row['web_url']) else None
        #             ),
        #             axis=1
        #         )

        # Split into batches
        for i, batch_start in enumerate(range(0, len(df), batch_size), start=1):
            batch_image_folder = image_dest_path / f"mini_batch{i}"
            batch_info_folder = info_dest_path / f"mini_batch{i}"

            batch_image_folder.mkdir(parents=True, exist_ok=True)
            batch_info_folder.mkdir(parents=True, exist_ok=True)

            batch_df = df.iloc[batch_start:batch_start + batch_size]
            batch_csv = batch_info_folder / f"mini_batch{i}.csv"
            batch_df.to_csv(batch_csv, index=False, encoding='utf-8')

            # Copy only the images for this specific batch
            missed_image_designs = {
                "fileid": [],
                "filename": []
            }
            for _, row in batch_df.iterrows():
                for image_name in [row["filename"]]:
                    if image_name != None:  # Skip if no image
                        image_path = source_path / image_name
                        if image_path.exists():
                            shutil.copy(image_path, batch_image_folder / image_name)
                        else:
                            missed_image_designs["fileid"].append(row["fileid"])
                            missed_image_designs["filename"].append(image_name)
            # Save missed images designs
            missed_batch_csv = batch_info_folder / f"missed_mini_batch{i}.csv"
            missed_batch_df = pd.DataFrame(missed_image_designs)
            missed_batch_df.to_csv(missed_batch_csv, index=False, encoding='utf-8')

        logger.success(f"Successfully split {len(df)} valid records into {((len(df) - 1) // batch_size) + 1} batches.")

    def setup(self):

        # Download data from GCS or not
        if self.config.data_download_enabled:
            # 1. Create local folder
            PipelineUtils.setup_directories(
                self.local_tmp_path, self.local_data_path
            )
            #
            # # 2. Download data
            # self.logger.info("Start downloading data...")
            # gcs = GCSUtils(self.config.bucket_name)
            # downloaded_files = gcs.download_folder(
            #     source_folder=self.gcs_source_folder,
            #     destination_folder=self.local_tmp_download_path
            # )
            # self.logger.info(f"gcs_source_folder: {self.gcs_source_folder}.")
            # self.logger.info(f"Successfully downloaded {downloaded_files}.")
            #
            # # get download file name
            # img_zip_file_name = FileUtils.find_files_by_ext(self.local_tmp_download_path, "zip")[0]
            # info_json_file_name = FileUtils.find_files_by_ext(self.local_tmp_download_path, "json")[0]
            # self.logger.info(f"Downloaded: {img_zip_file_name} and {info_json_file_name}")
            #
            # # 3. Unzip files
            # self.logger.info("Start unzipping data...")
            # extracted_files = FileUtils.unzip_files(
            #     zip_path=f"{self.local_tmp_download_path}/{img_zip_file_name}",
            #     output_dir=self.local_tmp_extract_path
            # )
            # self.logger.info(f"Successfully extracted {len(extracted_files)} images.")

            # 4. Get new designs information
            self.logger.info("Start getting design information...")
            new_designs_info_df = self.filter_new_designs_info(f"{self.local_tmp_download_path}/{info_json_file_name}")
            self.logger.info(f"Number of designs new processing designs: {new_designs_info_df.shape[0]}")

            # 5. Split new designs images and info
            self.logger.info("Start splitting data into smaller mini batches...")
            self.split_images_and_info_into_batches(
                batch_name=self.config.batch_name,
                source_folder=self.local_tmp_images_path,
                image_dest_folder=self.local_data_images_path,
                info_dest_folder=self.local_information_path,
                df=new_designs_info_df,
                batch_size=self.setup_data_config["mini_batch_size"],
                logger=self.logger
            )
        else:
            self.logger.warning("Setup data disabled.")


if __name__ == "__main__":
    config = Config()
    config.logger.info(f"Start setting up data...")
    SetupData(config).setup()