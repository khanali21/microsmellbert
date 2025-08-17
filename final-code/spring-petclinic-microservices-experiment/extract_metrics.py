"""
extract_metrics.py

This script extracts code metrics from a Java project using the CK tool and aggregates them per service.
Metrics include CBO, WMC, etc., saved to '<metrics_path>/metrics_{project}.csv' for use in model training.

Prerequisites:
- CK tool (ck.jar) in path from config.
- Java source files in '<repo_path>/<service>/src/main/java'.
- Libraries: os, subprocess, pandas, shutil.
- Configuration: config.json with 'repo_path', 'services', 'ck_jar', 'metric_cols'.

Usage: python extract_metrics.py
"""

import os
import subprocess
import pandas as pd
import shutil  # For temp dir handling
import json  # For loading config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)
repo_path = config['repo_path']
services = config.get('services', [])
project = config['project']
ck_jar = config.get('ck_jar', '../../ck/ck.jar')
metric_cols = config.get('metric_cols', ['cbo', 'wmc', 'dit', 'noc', 'rfc', 'lcom', 'loc', 'fanin', 'fanout', 'tcc', 'lcc'])
metrics_path = config['metrics_path']
logger.info(f"Loaded config for project: {project}")

def main():
    """
    Main function to extract and aggregate metrics using CK tool.

    Process:
    - Runs CK on each service's source code.
    - Aggregates means of configurable metrics.
    - Saves aggregated DataFrame to CSV.

    Returns:
        None: Saves CSV directly.
    """
    logger.info("Starting metrics extraction with CK.jar...")  # Debug start
    metrics_data = []

    # Get services dynamically
    if services == '*':
        services_list = [d for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)) and os.path.exists(os.path.join(repo_path, d, 'src/main/java'))]
        logger.info(f"Detected {len(services_list)} services using wildcard '*'")
    elif isinstance(services, str):  # Prefix filter
        services_list = [d for d in os.listdir(repo_path) if d.startswith(services) and os.path.isdir(os.path.join(repo_path, d)) and os.path.exists(os.path.join(repo_path, d, 'src/main/java'))]
        logger.info(f"Detected {len(services_list)} services using prefix '{services}'")
    else:
        services_list = [s for s in services if os.path.exists(os.path.join(repo_path, s, 'src/main/java'))]
        logger.info(f"Using {len(services_list)} specific services from config")

    for service in services_list:
        service_path = os.path.join(repo_path, service, 'src/main/java')
        logger.info(f"Running CK.jar on {service_path}...")  # Debug per service
        # Create per-service output dir
        output_dir = f'{metrics_path}/{service}/'
        os.makedirs(output_dir, exist_ok=True)
        try:
            # Run CK on source code (useJars=false for .java files)
            result = subprocess.run(['java', '-jar', ck_jar, 
                                     service_path, 'true', '0', 'true', output_dir], 
                                    capture_output=True, text=True, check=True)
            logger.info(f"CK.jar stdout sample: {result.stdout[:200]}...")  # Debug output
            if result.stderr:
                logger.warning(f"CK.jar stderr: {result.stderr}")  # Debug errors if any
            
            # Load class-level metrics from generated CSV
            class_csv = os.path.join(output_dir, 'class.csv')
            if not os.path.exists(class_csv):
                logger.warning(f"No class.csv generated for {service}; skipping.")
                continue
            
            df_class = pd.read_csv(class_csv)
            logger.info(f"Loaded {len(df_class)} classes for {service}. Columns: {df_class.columns.tolist()}")  # Debug load
            
            # Aggregate means for configurable columns + num_classes
            agg_metrics = {'service': service}
            for col in metric_cols:
                agg_metrics[f'{col}_mean'] = df_class[col].mean() if col in df_class else 0
            agg_metrics['num_classes'] = len(df_class)
            metrics_data.append(agg_metrics)
            logger.debug(f"Aggregated metrics for {service}: {agg_metrics}")  # Debug agg
            
            # Optional: Retain raw CSVs for reference; no cleanup here

        except subprocess.CalledProcessError as e:
            logger.error(f"CK.jar execution error for {service}: {e.stderr}")  # Debug subprocess error
        except Exception as e:
            logger.error(f"General error for {service}: {e}")

    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        metrics_csv = os.path.join(metrics_path, f'metrics_{project}.csv')
        df_metrics.to_csv(metrics_csv, index=False)
        logger.info(f"metrics_{project}.csv saved. Shape: {df_metrics.shape}")  # Debug save
    else:
        logger.warning("No metrics extracted; check CK.jar/paths.")

    logger.info("Metrics extraction complete.")  # Debug end

if __name__ == "__main__":
    main()