import os
import subprocess
import pandas as pd
import shutil  # For temp dir handling

def main():
    print("Starting metrics extraction with CK.jar...")  # Debug start
    repo_path = 'train-ticket'
    metrics_data = []
    ck_jar = '../ck/ck.jar'  # Assume in project root; adjust if ../ck/ck.jar

    # Create a temp output dir to avoid overwriting files
    temp_output = './temp_ck_output/'
    os.makedirs(temp_output, exist_ok=True)

    for service in os.listdir(repo_path):
        if not service.startswith('ts-'):
            print(f"Skipping non-service: {service}")  # Debug skip
            continue
        service_path = os.path.join(repo_path, service)
        if not os.path.exists(os.path.join(service_path, 'src/main/java')):
            print(f"No src/main/java found for {service}; skipping.")  # Debug missing
            continue
        
        print(f"Running CK.jar on {service_path}...")  # Debug per service
        try:
            # Run CK: <path> <useJars> <maxFiles> <printVars> <outputPath>
            # Use true for jars (if deps), 0 auto, true for vars/fields, temp_output
            result = subprocess.run(['java', '-jar', ck_jar, 
                                     service_path, 'true', '0', 'true', temp_output], 
                                    capture_output=True, text=True, check=True)
            print(f"CK.jar stdout sample: {result.stdout[:200]}...")  # Debug output
            print(f"CK.jar stderr: {result.stderr}")  # Debug errors if any
            
            # Read generated class.csv (class-level metrics)
            class_csv = os.path.join(temp_output, 'class.csv')
            if not os.path.exists(class_csv):
                print(f"No class.csv generated for {service}; skipping.")
                continue
            
            df_class = pd.read_csv(class_csv)
            print(f"Loaded {len(df_class)} classes for {service}. Columns: {df_class.columns.tolist()}")  # Debug load
            
            # Aggregate per service (e.g., mean of key metrics)
            agg_metrics = {
                'service': service,
                'cbo_mean': df_class['cbo'].mean() if 'cbo' in df_class else 0,
                'wmc_mean': df_class['wmc'].mean() if 'wmc' in df_class else 0,
                'dit_mean': df_class['dit'].mean() if 'dit' in df_class else 0,
                'noc_mean': df_class['noc'].mean() if 'noc' in df_class else 0,
                'rfc_mean': df_class['rfc'].mean() if 'rfc' in df_class else 0,
                'lcom_mean': df_class['lcom'].mean() if 'lcom' in df_class else 0,
                'loc_mean': df_class['loc'].mean() if 'loc' in df_class else 0,
                # Add more from your methodology: e.g., 'lcom*' if present, etc.
                # Full list from cat: cbo, cboModified, fanin, fanout, wmc, dit, noc, rfc, lcom, lcom*, tcc, lcc, etc.
                'fanin_mean': df_class['fanin'].mean() if 'fanin' in df_class else 0,
                'fanout_mean': df_class['fanout'].mean() if 'fanout' in df_class else 0,
                'tcc_mean': df_class['tcc'].mean() if 'tcc' in df_class else 0,
                'lcc_mean': df_class['lcc'].mean() if 'lcc' in df_class else 0,
                'num_classes': len(df_class)
            }
            metrics_data.append(agg_metrics)
            print(f"Aggregated metrics for {service}: {agg_metrics}")  # Debug agg
            
            # Optional: Clean temp files per service if needed; but we'll remove dir at end

        except subprocess.CalledProcessError as e:
            print(f"CK.jar execution error for {service}: {e.stderr}")  # Debug subprocess error
        except Exception as e:
            print(f"General error for {service}: {e}")

    # Clean up temp output
    shutil.rmtree(temp_output)
    print("Temp output cleaned.")  # Debug cleanup

    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv('metrics/metrics.csv', index=False)
        print("metrics.csv saved. Shape:", df_metrics.shape)  # Debug save
    else:
        print("No metrics extracted; check CK.jar/paths.")

    print("Metrics extraction complete.")  # Debug end

if __name__ == "__main__":
    main()