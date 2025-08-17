import os
import pandas as pd

def main():
    print("Starting script execution...")  # Initial debug

    # Load public dataset with error handling
    try:
        public_df = pd.read_csv('public_datasets/filtered_dataset_1.csv', 
                                engine='python', 
                                sep=',', 
                                quotechar='"', 
                                on_bad_lines='skip', 
                                encoding='utf-8')
        print("Public dataset loaded successfully.")  # Debug success
        print("Public dataset head:\n", public_df.head())  # Show first 5 rows
        print("Public dataset columns:", public_df.columns.tolist())  # Show columns
    except Exception as e:
        print(f"Error loading public dataset: {e}")  # Debug error
        public_df = pd.DataFrame()  # Fallback empty DF

    # For train-ticket: Heuristic initial labels
    data_path = 'data/services'
    print(f"Scanning services in: {data_path}")  # Debug path
    tt_labels = []
    for service in os.listdir(data_path):
        print(f"Processing service: {service}")  # Debug per service
        service_dir = os.path.join(data_path, service)
        if not os.path.isdir(service_dir):
            print(f"Skipping non-dir: {service_dir}")
            continue
        loc = 0
        num_classes = len([f for f in os.listdir(service_dir) if f.endswith('.java')])
        print(f"  Num classes: {num_classes}")  # Debug count
        for file in os.listdir(service_dir):
            if file.endswith('.java'):
                file_path = os.path.join(service_dir, file)
                try:
                    with open(file_path, 'r') as f:
                        loc += sum(1 for _ in f)
                    print(f"    Processed file: {file}, LOC added: {loc}")  # Debug per file
                except Exception as e:
                    print(f"    Error reading {file}: {e}")
        # Heuristics (tweak for your methodology)
        mega = 1 if loc > 1000 or num_classes > 15 else 0
        # CRUDy: Basic check for 'Mapping' (e.g., Spring annotations)
        crudy_check_file = next((f for f in os.listdir(service_dir) if f.endswith('.java')), None)
        crudy = 0
        if crudy_check_file:
            try:
                with open(os.path.join(service_dir, crudy_check_file), 'r') as f:
                    content = f.read()
                    crudy = 1 if 'Mapping' in content else 0
                print(f"  CRUDy check on {crudy_check_file}: {crudy}")  # Debug CRUDy
            except Exception as e:
                print(f"  Error checking CRUDy: {e}")
        ambiguous = 0  # Manual or advanced (e.g., coherence via embeddings later)
        tt_labels.append({'service': service, 'mega_service': mega, 'crudy_service': crudy, 'ambiguous_service': ambiguous})
        print(f"  Labels for {service}: mega={mega}, crudy={crudy}, ambiguous={ambiguous}")  # Debug labels

    # Combine: Use public for training, train-ticket for testing
    tt_df = pd.DataFrame(tt_labels)
    print("Train-ticket DF head:\n", tt_df.head())  # Debug tt_df

    if not public_df.empty:
        try:
            # Map public columns (example; adjust based on actual headers from debug)
            # Assuming sample columns; replace with real ones after seeing print(public_df.columns)
            public_df = public_df.rename(columns={'Project': 'service', 'Hub-like': 'mega_service', 'CRUD Service': 'crudy_service', 'Ambiguous Service': 'ambiguous_service'})
            print("Public DF renamed columns:", public_df.columns.tolist())  # Debug rename
            combined_df = pd.concat([public_df[['service', 'mega_service', 'crudy_service', 'ambiguous_service']], tt_df])
            print("Combined DF shape:", combined_df.shape)  # Debug shape
        except KeyError as e:
            print(f"Column mapping error: {e}. Using only train-ticket data.")
            combined_df = tt_df
    else:
        combined_df = tt_df

    try:
        combined_df.to_csv('labels.csv', index=False)
        print("labels.csv saved successfully.")  # Debug save
    except Exception as e:
        print(f"Error saving labels.csv: {e}")

    print("Script execution complete.")  # Final debug

if __name__ == "__main__":
    main()