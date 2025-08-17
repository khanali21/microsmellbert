"""
extract_embeddings.py

This script extracts semantic embeddings from preprocessed code (.pre files) in a Java project using CodeBERT.
Embeddings are service-level and saved as a pickle file in the configured embeddings path.

Prerequisites:
- Preprocessed '.pre' files in '<data_path>/services/<service>/'.
- Libraries: torch, transformers, pandas, numpy.
- Configuration: config.json with 'data_path', 'services', 'embeddings_path', 'project'.

Usage: python extract_embeddings.py
"""

import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import numpy as np
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
embeddings_path = config['embeddings_path']
project = config['project']
logger.info(f"Loaded config for project: {project}")

def main():
    """
    Main function to extract and save CodeBERT embeddings.

    Process:
    - Concatenates preprocessed code per service.
    - Generates 768D embeddings using CodeBERT (pooler_output).
    - Saves as DataFrame in pickle format.

    Returns:
        None: Saves pickle directly.
    """
    logger.info("Starting embeddings extraction...")  # Debug start
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")  # Debug device

    # Load CodeBERT model and tokenizer (pre-trained for code semantics)
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = RobertaModel.from_pretrained('microsoft/codebert-base').to(device)
    logger.info("CodeBERT loaded.")  # Debug model load

    embeddings = []
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

    for service in services_list:
        logger.info(f"Processing service: {service}")  # Debug per service
        service_dir = os.path.join(data_path, service)
        service_code = ''
        pre_files = [f for f in os.listdir(service_dir) if f.endswith('.pre')]
        if not pre_files:
            logger.warning(f"No preprocessed files for {service}; skipping.")
            continue
        # Concatenate all preprocessed files for service-level embedding
        for file in pre_files:
            file_path = os.path.join(service_dir, file)
            with open(file_path, 'r') as f:
                service_code += f.read() + '\n'
            logger.debug(f"  Added file: {file}, code length: {len(service_code)}")  # Debug code buildup
        
        if len(service_code.strip()) == 0:
            logger.warning(f"Empty code for {service}; skipping.")
            continue
        
        try:
            # Tokenize and truncate to max_length for model efficiency
            inputs = tokenizer(service_code, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # Use pooler_output for fixed-size service embedding
            emb = outputs.pooler_output.cpu().detach().numpy()  # 768D vector
            logger.debug(f"  Embedding shape: {emb.shape}")  # Debug emb
            embeddings.append({'service': service, 'embedding': emb.flatten()})
        except Exception as e:
            logger.error(f"Embedding error for {service}: {e}")

    if embeddings:
        os.makedirs(embeddings_path, exist_ok=True)
        df_emb = pd.DataFrame(embeddings)
        embeddings_pkl = os.path.join(embeddings_path, f'embeddings_{project}.pkl')
        df_emb.to_pickle(embeddings_pkl)  # Pickle for numpy arrays
        logger.info(f"embeddings_{project}.pkl saved. Num services: {len(df_emb)}")  # Debug save
    else:
        logger.warning("No embeddings extracted; check preprocessing.")

    logger.info("Embeddings extraction complete.")  # Debug end

if __name__ == "__main__":
    main()