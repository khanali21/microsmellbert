"""
extract_metrics_petclinic.py

This script extracts code metrics from Spring PetClinic microservices using the CK tool and aggregates them per service.
Metrics include CBO, WMC, etc., saved to 'metrics/metrics_petclinic.csv' for use in model training.

Prerequisites:
- CK tool (ck.jar) in '../ck/ck.jar' or adjust path.
- Java source files in 'petclinic-microservices/<service>/src/main/java'.
- Libraries: os, subprocess, pandas, shutil.

Usage: python extract_metrics_petclinic.py
"""

import os
import subprocess
import pandas as pd
import shutil  # For temp dir handling

def main():
    """
    Main function to extract and aggregate metrics using CK tool.

    Process:
    - Runs CK on each service's source code.
    - Aggregates means of key metrics (e.g., cbo_mean).
    - Saves aggregated DataFrame to CSV.

    Returns:
        None: Saves CSV directly.
    """
    print("Starting metrics extraction with CK.jar...")  # Debug start
    repo_path = 'petclinic-microservices'
    metrics_data = []
    ck_jar = '../ck/ck.jar'  # Path to CK JAR; ensure it's downloadable/executable

    # List of services to process
    services = [
        'spring-petclinic-api-gateway',
        'spring-petclinic-customers-service',
        'spring-petclinic-vets-service',
        'spring-petclinic-visits-service',
        'spring-petclinic-admin-server',
        'spring-petclinic-config-server',
        'spring-petclinic-discovery-server'
    ]

    for service in services:
        service_path = os.path.join(repo_path, service, 'src/main/java')
        if not os.path.exists(service_path):
            print(f"No src/main/java found for {service}; skipping.")  # Debug missing
            continue
        
        print(f"Running CK.jar on {service_path}...")  # Debug per service
        # Create per-service output dir to store CK results
        output_dir = f'metrics/{service}'
        os.makedirs(output_dir, exist_ok=True)
        try:
            # Run CK on source code (useJars=false for .java files)
            result = subprocess.run(['java', '-jar', ck_jar, 
                                     service_path, 'false', '0', 'true', output_dir], 
                                    capture_output=True, text=True, check=True)
            print(f"CK.jar stdout sample: {result.stdout[:200]}...")  # Debug output
            print(f"CK.jar stderr: {result.stderr}")  # Debug errors if any
            
            # Load class-level metrics from generated CSV
            class_csv = os.path.join(output_dir, 'class.csv')
            if not os.path.exists(class_csv):
                print(f"No class.csv generated for {service}; skipping.")
                continue
            
            df_class = pd.read_csv(class_csv)
            print(f"Loaded {len(df_class)} classes for {service}. Columns: {df_class.columns.tolist()}")  # Debug load
            
            # Aggregate means for key metrics (handles missing columns gracefully)
            agg_metrics = {
                'service': service,
                'cbo_mean': df_class['cbo'].mean() if 'cbo' in df_class else 0,
                'wmc_mean': df_class['wmc'].mean() if 'wmc' in df_class else 0,
                'dit_mean': df_class['dit'].mean() if 'dit' in df_class else 0,
                'noc_mean': df_class['noc'].mean() if 'noc' in df_class else 0,
                'rfc_mean': df_class['rfc'].mean() if 'rfc' in df_class else 0,
                'lcom_mean': df_class['lcom'].mean() if 'lcom' in df_class else 0,
                'loc_mean': df_class['loc'].mean() if 'loc' in df_class else 0,
                'fanin_mean': df_class['fanin'].mean() if 'fanin' in df_class else 0,
                'fanout_mean': df_class['fanout'].mean() if 'fanout' in df_class else 0,
                'tcc_mean': df_class['tcc'].mean() if 'tcc' in df_class else 0,
                'lcc_mean': df_class['lcc'].mean() if 'lcc' in df_class else 0,
                'num_classes': len(df_class)
            }
            metrics_data.append(agg_metrics)
            print(f"Aggregated metrics for {service}: {agg_metrics}")  # Debug agg
            
            # Optional: Retain raw CSVs for reference; no cleanup here

        except subprocess.CalledProcessError as e:
            print(f"CK.jar execution error for {service}: {e.stderr}")  # Debug subprocess error
        except Exception as e:
            print(f"General error for {service}: {e}")

    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv('metrics/metrics_petclinic.csv', index=False)
        print("metrics_petclinic.csv saved. Shape:", df_metrics.shape)  # Debug save
    else:
        print("No metrics extracted; check CK.jar/paths.")

    print("Metrics extraction complete.")  # Debug end

if __name__ == "__main__":
    main()