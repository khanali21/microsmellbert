"""
generate_labels.py

This script generates heuristic-based labels for anti-patterns (Mega-Service, CRUDy Service, Ambiguous Service) in a Java project.
It uses simple thresholds (e.g., LOC, class count) and augments with public datasets if available. Outputs to '<data_path>/labels_{project}.csv'.

Prerequisites:
- Java files in '<data_path>/services/<service>/'.
- Optional: Public dataset CSV in 'public_datasets/' for augmentation.
- Configuration: config.json with 'data_path', 'services', 'crudy_annotations', 'mega_loc_threshold', 'mega_classes_threshold'.

Usage: python generate_labels.py
"""

import os
import pandas as pd
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
project = config['project']
crudy_annotations = config.get('crudy_annotations', ['@GetMapping', '@PostMapping'])
mega_loc_threshold = config.get('mega_loc_threshold', 500)
mega_classes_threshold = config.get('mega_classes_threshold', 10)
logger.info(f"Loaded config for project: {project}, thresholds: LOC={mega_loc_threshold}, classes={mega_classes_threshold}")

def is_mega_service(num_classes, loc_mean):
    """
    Determine if a service is a Mega-Service based on configurable thresholds.

    Args:
        num_classes (int): Number of classes in the service.
        loc_mean (float): Average lines of code per class.

    Returns:
        int: 1 if Mega-Service, 0 otherwise.
    """
    return 1 if num_classes > mega_classes_threshold or loc_mean > mega_loc_threshold else 0

def is_crudy_service(operations):
    """
    Determine if a service is CRUDy based on operation types (simplified heuristic).

    Args:
        operations (list): List of operation types (e.g., from annotations).

    Returns:
        int: 1 if CRUDy, 0 otherwise (placeholder; requires parsing improvement).
    """
    crud_count = sum(1 for op in operations if any(ann in op for ann in crudy_annotations))
    return 1 if crud_count > 0.7 * len(operations) else 0

def is_ambiguous_service(responsibilities):
    """
    Determine if a service is Ambiguous (placeholder heuristic).

    Args:
        responsibilities (list): List of responsibilities (not fully implemented).

    Returns:
        int: 1 if Ambiguous, 0 otherwise (placeholder).
    """
    return 1 if len(responsibilities) > 3 else 0

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

# Collect data per service
labels = []
for service in services_list:
    service_path = os.path.join(data_path, service)
    if os.path.isdir(service_path):
        num_classes = len([f for f in os.listdir(service_path) if f.endswith('.java')])
        loc_mean = sum(len(open(os.path.join(service_path, f)).readlines()) for f in os.listdir(service_path) if f.endswith('.java')) / num_classes if num_classes > 0 else 0
        # Placeholder for operations and responsibilities (requires parsing enhancement)
        operations = []  # TODO: Parse annotations like @GetMapping
        responsibilities = []  # TODO: Semantic analysis
        
        mega = is_mega_service(num_classes, loc_mean)
        crudy = is_crudy_service(operations)
        ambiguous = is_ambiguous_service(responsibilities)
        
        labels.append([service, mega, crudy, ambiguous])
        logger.debug(f"Generated labels for {service}: Mega={mega}, CRUDy={crudy}, Ambiguous={ambiguous}")

# Save to CSV
df = pd.DataFrame(labels, columns=['Service', 'Mega-Service', 'CRUDy Service', 'Ambiguous Service'])
labels_csv = os.path.join(config['data_path'], f'labels_{project}.csv')
df.to_csv(labels_csv, index=False)
logger.info(f"Labels generated: {labels_csv}, shape: {df.shape}")