"""
preprocess_code_petclinic.py

This script preprocesses Java source files from the Spring PetClinic microservices by removing comments, normalizing variable names,
and excluding auto-generated files. The output is saved as '.pre' files for use in semantic embedding extraction.

Prerequisites:
- Java files extracted into 'data/services/<service>/' directories.
- Libraries: re, javalang (for potential advanced parsing, though basic regex is used here).

Usage: python preprocess_code_petclinic.py
"""

import os
import re
import javalang  # For better parsing if needed (not actively used in basic implementation)

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

# Define base path for preprocessed data
data_path = 'data/services'

# List of PetClinic services to process (hardcoded based on repo structure for consistency)
services = [
    'spring-petclinic-api-gateway',
    'spring-petclinic-customers-service',
    'spring-petclinic-vets-service',
    'spring-petclinic-visits-service',
    'spring-petclinic-admin-server',
    'spring-petclinic-config-server',
    'spring-petclinic-discovery-server'
]

# Process each service directory
for service in services:
    service_dir = os.path.join(data_path, service)
    if not os.path.exists(service_dir):
        print(f"Skipping missing dir: {service_dir}")
        continue
    # Iterate over Java files in the service directory
    for file in os.listdir(service_dir):
        if file.endswith('.java'):
            file_path = os.path.join(service_dir, file)
            preprocessed = preprocess_code(file_path)
            if preprocessed:  # Only save if not skipped (e.g., not auto-generated)
                with open(file_path + '.pre', 'w') as f:
                    f.write(preprocessed)

print("Preprocessing complete.")