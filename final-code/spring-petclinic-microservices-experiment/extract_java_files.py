"""
extract_java_files.py

This script extracts Java source files from a Java project repository and organizes them into service directories
under the configured data path.

Prerequisites:
- Repository cloned at '<repo_path>'.
- Configuration: config.json with 'repo_path', 'data_path', 'services'.
- Libraries: os, shutil.

Usage: python extract_java_files.py
"""

import os
import shutil
import json  # For loading config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)
repo_path = config['repo_path']
data_path = os.path.join(config['data_path'], 'services')
services = config.get('services', [])

# Create output dirs and copy .java files
if services == '*':
    services_list = [d for d in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, d)) and os.path.exists(os.path.join(repo_path, d, 'src/main/java'))]
    logger.info(f"Detected {len(services_list)} services using wildcard '*'")
elif isinstance(services, str):  # Prefix filter, e.g., 'ts-'
    services_list = [d for d in os.listdir(repo_path) if d.startswith(services) and os.path.isdir(os.path.join(repo_path, d)) and os.path.exists(os.path.join(repo_path, d, 'src/main/java'))]
    logger.info(f"Detected {len(services_list)} services using prefix '{services}'")
else:  # List of specific services
    services_list = [s for s in services if os.path.exists(os.path.join(repo_path, s, 'src/main/java'))]
    logger.info(f"Using {len(services_list)} specific services from config")

for service in services_list:
    src_dir = os.path.join(repo_path, service, 'src/main/java')
    dest_dir = os.path.join(data_path, service)
    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"Processing service: {service}, source: {src_dir}, destination: {dest_dir}")
    
    if os.path.exists(src_dir):
        java_files_count = 0
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.java'):
                    shutil.copy(os.path.join(root, file), dest_dir)
                    java_files_count += 1
        logger.info(f"Copied {java_files_count} .java files for service {service}")
    else:
        logger.warning(f"{src_dir} not found for {service}")

logger.info("Java files extraction complete.")