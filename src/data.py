import os
import pandas as pd
import xarray as xr
import numpy as np
from urllib.parse import urljoin, urlparse
from io import BytesIO
import mysql.connector
import gzip
from ftplib import FTP
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import concurrent.futures
from dataclasses import dataclass
import json
from pathlib import Path

# --------------------
# CONFIGURATION AND LOGGING SETUP
# --------------------
@dataclass
class ArgoConfig:
    """Configuration class for ARGO data processing"""
    lat_min: float = -5
    lat_max: float = 5
    lon_min: float = 60
    lon_max: float = 80
    start_date: str = "2023-03-01"
    end_date: str = "2023-03-31"
    ftp_base_url: str = "ftp://ftp.ifremer.fr/ifremer/argo/"
    download_folder: str = "argo_data_downloads"
    batch_size: int = 10
    max_workers: int = 4
    vars_to_keep: List[str] = None
    
    def __post_init__(self):
        if self.vars_to_keep is None:
            self.vars_to_keep = ["TEMP", "PSAL", "DOXY"]

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('argo_ingestion.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

config = ArgoConfig()
logger = setup_logging()
os.makedirs(config.download_folder, exist_ok=True)

# --------------------
# ENHANCED DATA INGESTION CLASS
# --------------------
class ArgoDataIngestion:
    """Enhanced ARGO data ingestion system with improved error handling and scalability"""
    
    def __init__(self, config: ArgoConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.index_file_url = urljoin(config.ftp_base_url, "ar_index_global_prof.txt.gz")
    
    def download_file_ftp(self, file_url: str, dest_path: str) -> bool:
        """Downloads a single file from an FTP URL with enhanced error handling."""
        try:
            url_parts = urlparse(file_url)
            ftp_server = url_parts.netloc
            ftp_path = os.path.dirname(url_parts.path).lstrip('/')
            file_name = os.path.basename(url_parts.path)

            with FTP(ftp_server) as ftp:
                ftp.login()
                ftp.cwd(ftp_path)
                with open(dest_path, "wb") as f:
                    ftp.retrbinary(f"RETR {file_name}", f.write)
            
            self.logger.info(f"Successfully downloaded: {file_name}")
            return True
        except Exception as e:
            self.logger.error(f"FTP Download Failed for {file_url}: {e}")
            return False
    
    def fetch_global_index(self) -> pd.DataFrame:
        """Fetch and parse the global ARGO index."""
        self.logger.info(f"Fetching global ARGO index via FTP: {self.index_file_url}")
        
        url_parts = urlparse(self.index_file_url)
        ftp_server = url_parts.netloc
        ftp_path = os.path.dirname(url_parts.path).lstrip('/')
        file_name = os.path.basename(url_parts.path)
        ftp_file_buffer = BytesIO()

        try:
            with FTP(ftp_server) as ftp:
                ftp.login()
                ftp.cwd(ftp_path)
                ftp.retrbinary(f"RETR {file_name}", ftp_file_buffer.write)
            
            ftp_file_buffer.seek(0)
            with gzip.open(ftp_file_buffer, 'rt') as f:
                df_index = pd.read_csv(f, comment='#', header=0)
            
            self.logger.info(f"Successfully loaded {len(df_index)} profiles from index")
            return df_index
            
        except Exception as e:
            self.logger.error(f"Error during FTP index download: {e}")
            raise
    
    def filter_profiles(self, df_index: pd.DataFrame) -> pd.DataFrame:
        """Filter profiles based on geographic and temporal criteria."""
        df_index['date'] = pd.to_datetime(df_index['date'], format='%Y%m%d%H%M%S')
        
        self.logger.info(f"Filtering {len(df_index)} total profiles...")
        filtered = df_index[
            (df_index['latitude'] >= self.config.lat_min) & 
            (df_index['latitude'] <= self.config.lat_max) &
            (df_index['longitude'] >= self.config.lon_min) & 
            (df_index['longitude'] <= self.config.lon_max) &
            (df_index['date'] >= self.config.start_date) & 
            (df_index['date'] <= self.config.end_date)
        ].copy()
        
        self.logger.info(f"Found {len(filtered)} profiles matching criteria")
        return filtered
    
    def download_profiles_batch(self, profiles: pd.DataFrame) -> List[str]:
        """Download profile files in batches with concurrent processing."""
        downloaded_files = []
        
        def download_single_profile(row) -> Optional[str]:
            file_path = row['file']
            fname = os.path.basename(file_path)
            file_url = urljoin(self.config.ftp_base_url, file_path)
            dest = os.path.join(self.config.download_folder, fname)

            if not os.path.exists(dest):
                if self.download_file_ftp(file_url, dest):
                    return dest
                return None
            else:
                self.logger.info(f"Already exists: {fname}")
                return dest
        
        # Process in batches to avoid overwhelming the FTP server
        for i in range(0, len(profiles), self.config.batch_size):
            batch = profiles.iloc[i:i + self.config.batch_size]
            self.logger.info(f"Processing batch {i//self.config.batch_size + 1}")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                batch_results = list(executor.map(download_single_profile, 
                                                [row for _, row in batch.iterrows()]))
            
            downloaded_files.extend([f for f in batch_results if f is not None])
        
        return downloaded_files

# --------------------
# STEP D: FETCH AND PARSE THE GLOBAL INDEX (UPDATED)
# --------------------
ingestion_system = ArgoDataIngestion(config)
df_index = ingestion_system.fetch_global_index()
filtered = ingestion_system.filter_profiles(df_index)

# --------------------
# STEP E: DOWNLOAD DATA FILES (UPDATED)
# --------------------
downloaded_files = ingestion_system.download_profiles_batch(filtered)

# --------------------
# ENHANCED DATA PROCESSING
# --------------------
class ArgoDataProcessor:
    """Enhanced processor for ARGO NetCDF files"""
    
    def __init__(self, config: ArgoConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_nc_file(self, nc_path: str) -> pd.DataFrame:
        """Process a single NetCDF file with enhanced error handling and data validation."""
        try:
            ds = xr.open_dataset(nc_path)
            self.logger.info(f"Processing {os.path.basename(nc_path)}")
        except Exception as e:
            self.logger.error(f"Error opening {nc_path}: {e}")
            return pd.DataFrame()

        try:
            # Extract metadata
            float_id = str(ds["PLATFORM_NUMBER"].values.item())
            times = pd.to_datetime(ds["JULD"].values, origin="1950-01-01", unit="D")
            lat = ds["LATITUDE"].values
            lon = ds["LONGITUDE"].values
            depth = ds["PRES"].values

            rows = []
            for i in range(ds.dims['N_PROF']):
                for j in range(ds.dims['N_LEVELS']):
                    # Skip invalid depth values
                    if np.isnan(depth[i, j]) or depth[i, j] < 0:
                        continue
                    
                    record = {
                        "float_id": float_id,
                        "date": times[i],
                        "latitude": float(lat[i]),
                        "longitude": float(lon[i]),
                        "depth": float(depth[i, j]),
                        "profile_index": i,
                        "level_index": j
                    }
                    
                    # Add oceanographic variables with quality control
                    valid_vars = 0
                    for var in self.config.vars_to_keep:
                        if var in ds:
                            qc_var = var + "_QC"
                            value = ds[var].values[i, j]
                            
                            # Check quality control if available
                            if qc_var in ds:
                                qc_flag = ds[qc_var].values[i, j]
                                if qc_flag in [b'1', b'2']:  # Good or probably good data
                                    if not np.isnan(value):
                                        record[var.lower()] = float(value)
                                        valid_vars += 1
                            else:
                                # No QC available, just check for valid values
                                if not np.isnan(value):
                                    record[var.lower()] = float(value)
                                    valid_vars += 1
                    
                    # Only keep records with at least one valid oceanographic variable
                    if valid_vars > 0:
                        rows.append(record)

            df = pd.DataFrame(rows)
            self.logger.info(f"Extracted {len(df)} valid measurements from {os.path.basename(nc_path)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing data from {nc_path}: {e}")
            return pd.DataFrame()
        finally:
            ds.close()
    
    def process_all_files(self, file_paths: List[str]) -> pd.DataFrame:
        """Process all NetCDF files and combine into a single DataFrame."""
        all_dataframes = []
        
        for fpath in file_paths:
            df = self.process_nc_file(fpath)
            if not df.empty:
                all_dataframes.append(df)
        
        if not all_dataframes:
            self.logger.warning("No data was successfully processed")
            return pd.DataFrame()
        
        combined = pd.concat(all_dataframes, ignore_index=True)
        
        # Rename columns for consistency
        column_mapping = {
            "temp": "temperature",
            "psal": "salinity", 
            "doxy": "dissolved_oxygen"
        }
        combined.rename(columns=column_mapping, inplace=True)
        
        self.logger.info(f"Final combined dataset shape: {combined.shape}")
        return combined

# --------------------
# STEP F: PROCESS DOWNLOADED FILES (UPDATED)
# --------------------
processor = ArgoDataProcessor(config)
combined = processor.process_all_files(downloaded_files)
# --------------------
# ENHANCED DATABASE OPERATIONS
# --------------------
class ArgoDatabase:
    """Enhanced database operations for ARGO data"""
    
    def __init__(self, config: ArgoConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection_params = {
            "host": "localhost",
            "user": "root", 
            "password": "Arman123?",
            "database": "argo_data",
            "allow_local_infile": True
        }
    
    def create_enhanced_schema(self):
        """Create enhanced database schema with additional indexes and metadata tables."""
        try:
            conn = mysql.connector.connect(**self.connection_params)
            cursor = conn.cursor()
            
            # Main profiles table with enhanced schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS argo_profiles (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    float_id VARCHAR(50) NOT NULL,
                    date DATETIME NOT NULL,
                    latitude DOUBLE NOT NULL,
                    longitude DOUBLE NOT NULL,
                    depth DOUBLE NOT NULL,
                    temperature DOUBLE NULL,
                    salinity DOUBLE NULL,
                    dissolved_oxygen DOUBLE NULL,
                    profile_index INT NULL,
                    level_index INT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    
                    INDEX idx_date (date),
                    INDEX idx_latlon (latitude, longitude),
                    INDEX idx_float (float_id),
                    INDEX idx_depth (depth),
                    INDEX idx_location_date (latitude, longitude, date),
                    INDEX idx_float_date (float_id, date)
                ) ENGINE=InnoDB;
            """)
            
            # Metadata table for processing runs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_metadata (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    run_id VARCHAR(100) UNIQUE NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NULL,
                    status ENUM('running', 'completed', 'failed') DEFAULT 'running',
                    profiles_processed INT DEFAULT 0,
                    files_processed INT DEFAULT 0,
                    config_used JSON,
                    error_message TEXT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB;
            """)
            
            conn.commit()
            self.logger.info("Database schema created successfully")
            
        except mysql.connector.Error as err:
            self.logger.error(f"Error creating database schema: {err}")
            raise
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()
    
    def save_to_database(self, df: pd.DataFrame, run_id: str) -> bool:
        """Save processed data to database with metadata tracking."""
        if df.empty:
            self.logger.warning("No data to save to database")
            return False
        
        try:
            conn = mysql.connector.connect(**self.connection_params)
            cursor = conn.cursor()
            
            # Create schema if it doesn't exist
            self.create_enhanced_schema()
            
            # Start processing run metadata
            cursor.execute("""
                INSERT INTO processing_metadata (run_id, start_time, config_used)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                start_time = VALUES(start_time), config_used = VALUES(config_used)
            """, (run_id, datetime.now(), json.dumps(self.config.__dict__, default=str)))
            
            # Save data to CSV for bulk loading
            csv_path = f"argo_processed_{run_id}.csv"
            df.to_csv(csv_path, index=False)
            
            # Clear existing data (optional - comment out for incremental loading)
            cursor.execute("TRUNCATE TABLE argo_profiles;")
            self.logger.info("Cleared existing data from argo_profiles table")
            
            # Bulk load data
            csv_full_path = os.path.abspath(csv_path).replace("\\", "/")
            
            # Determine columns based on what's available in the dataframe
            available_columns = df.columns.tolist()
            db_columns = ["float_id", "date", "latitude", "longitude", "depth"]
            optional_columns = ["temperature", "salinity", "dissolved_oxygen", "profile_index", "level_index"]
            
            # Only include optional columns that exist in the dataframe
            for col in optional_columns:
                if col in available_columns:
                    db_columns.append(col)
            
            columns_str = ", ".join(db_columns)
            
            load_sql = f"""
                LOAD DATA LOCAL INFILE '{csv_full_path}' 
                INTO TABLE argo_profiles 
                FIELDS TERMINATED BY ',' 
                ENCLOSED BY '"' 
                LINES TERMINATED BY '\\n' 
                IGNORE 1 ROWS 
                ({columns_str});
            """
            
            cursor.execute(load_sql)
            rows_loaded = cursor.rowcount
            
            # Update metadata
            cursor.execute("""
                UPDATE processing_metadata 
                SET end_time = %s, status = 'completed', profiles_processed = %s
                WHERE run_id = %s
            """, (datetime.now(), rows_loaded, run_id))
            
            conn.commit()
            self.logger.info(f"Successfully loaded {rows_loaded} rows into MySQL")
            
            # Clean up CSV file
            os.remove(csv_path)
            return True
            
        except mysql.connector.Error as err:
            self.logger.error(f"Error loading data into MySQL: {err}")
            # Update metadata with error
            try:
                cursor.execute("""
                    UPDATE processing_metadata 
                    SET end_time = %s, status = 'failed', error_message = %s
                    WHERE run_id = %s
                """, (datetime.now(), str(err), run_id))
                conn.commit()
            except:
                pass
            return False
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

# --------------------
# STEP G: SAVE TO DATABASE (UPDATED)
# --------------------
if not combined.empty:
    run_id = f"argo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    database = ArgoDatabase(config)
    success = database.save_to_database(combined, run_id)
    
    if success:
        logger.info("Data ingestion completed successfully")
    else:
        logger.error("Data ingestion failed")
else:
    logger.warning("No data was processed - skipping database save")