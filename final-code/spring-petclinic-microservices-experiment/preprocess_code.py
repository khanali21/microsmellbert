"""
preprocess_code.py

This script preprocesses Java source files from a Java project by removing comments, normalizing variable names,
and excluding auto-generated files. The output is saved as '.pre' files in the configured data path.

Prerequisites:
- Java files extracted into '<data_path>/services/<service>/' directories (from config).
- Libraries: re, javalang (for potential advanced parsing, though basic regex is used here).
- Configuration: config.json with 'data_path' and 'services'.

Usage: python preprocess_code.py
"""

import os
import re
import javalang  # For better parsing if needed (not actively used in basic implementation)
import json  # For loading config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)
data_path = os.path.join(config['data_path'], 'services')
services = config.get('services', [])
logger.info("Loaded config")

def preprocess_code(file_path):
    """
    Preprocess a single Java file by removing comments, normalizing identifiers to 'VAR', and skipping auto-generated files.

    Args:
        file_path (str): Path to the input .java file.

    Returns:
        str or None: Preprocessed code string, or None if the file is auto-generated.
    """
    with open(file_path, 'r') as f:
        code = f.read()
    
    # Remove comments using regex to handle single-line and multi-line comments efficiently
    code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', code)
    
    # Normalize variables/identifiers to 'VAR' for semantic abstraction (reduces noise in embeddings)
    code = re.sub(r'\b[A-Za-z_]\w*\b', 'VAR', code)
    
    # Exclude auto-generated files by checking for '@Generated' annotation to avoid irrelevant code
    if '@Generated' in code:
        return None  # Skip file
    
    return code

# Get services dynamically based on config
if services == '*':
    services_list = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    logger.info(f"Detected {len(services_list)} services using wildcard '*'")
elif isinstance(services, str):  # Prefix filter, e.g., 'ts-'
    services_list = [d for d in os.listdir(data_path) if d.startswith(services) and os.path.isdir(os.path.join(data_path, d))]
    logger.info(f"Detected {len(services_list)} services using prefix '{services}'")
else:  # List of specific services
    services_list = services
    logger.info(f"Using {len(services_list)} specific services from config")

# Process each service directory
for service in services_list:
    service_dir = os.path.join(data_path, service)
    if not os.path.exists(service_dir):
        logger.warning(f"Skipping missing dir: {service_dir}")
        continue
    # Iterate over Java files in the service directory
    java_files_count = 0
    pre_files_count = 0
    for file in os.listdir(service_dir):
        if file.endswith('.java'):
            java_files_count += 1
            file_path = os.path.join(service_dir, file)
            preprocessed = preprocess_code(file_path)
            if preprocessed:  # Only save if not skipped (e.g., not auto-generated)
                pre_file_path = file_path + '.pre'
                with open(pre_file_path, 'w') as f:
                    f.write(preprocessed)
                pre_files_count += 1
                logger.debug(f"Preprocessed {file} to {pre_file_path}")
            else:
                logger.debug(f"Skipped auto-generated file: {file}")
    logger.info(f"Processed service {service}: {java_files_count} .java files, {pre_files_count} .pre files created")

logger.info("Preprocessing complete.")