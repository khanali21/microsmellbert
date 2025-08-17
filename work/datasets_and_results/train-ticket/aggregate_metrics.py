import os
import csv
import glob

def parse_arff_file(arff_path):
    """Parse an ARFF file and count smell occurrences."""
    smells = {}
    data_section = False
    
    if not os.path.exists(arff_path):
        return smells
        
    with open(arff_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.lower().startswith('@data'):
                data_section = True
                continue
                
            if data_section:
                smell_type = line.split(',')[-1].strip()
                smells[smell_type] = smells.get(smell_type, 0) + 1
                
    return smells

def parse_csv_file(csv_path):
    """Count number of rows in a CSV file (excluding header)."""
    if not os.path.exists(csv_path):
        return 0
        
    with open(csv_path, 'r') as f:
        return sum(1 for _ in csv.reader(f)) - 1

def main():
    # Configuration
    output_csv = 'aggregated_metrics.csv'
    all_smell_types = set()
    aggregated_data = []

    # Find all service directories
    service_dirs = [d for d in os.listdir() 
                   if os.path.isdir(d) and d.startswith('ts-')]
    
    # Process each service directory
    for service in service_dirs:
        print(f"Processing {service}...")
        
        # Initialize service data
        service_data = {
            'service': service,
            'loc': 0,
            'methods': 0,
            'classes': 0
        }
        
        # Process class.csv for LOC and class count
        class_csv = os.path.join(service, 'class.csv')
        if os.path.exists(class_csv):
            with open(class_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    service_data['classes'] += 1
                    try:
                        service_data['loc'] += int(row.get('loc', 0))
                    except ValueError:
                        pass
        
        # Process method.csv for method count
        method_csv = os.path.join(service, 'method.csv')
        service_data['methods'] = parse_csv_file(method_csv)
        
        # Process ARFF file for smell counts
        arff_files = glob.glob(os.path.join(service, '*_smells.arff'))
        if arff_files:
            smells = parse_arff_file(arff_files[0])
            for smell, count in smells.items():
                service_data[smell] = count
                all_smell_types.add(smell)
        
        aggregated_data.append(service_data)
    
    # Write aggregated results to CSV
    fieldnames = ['service', 'loc', 'methods', 'classes'] + sorted(all_smell_types)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for data in aggregated_data:
            # Fill in missing smells with 0
            for smell in all_smell_types:
                if smell not in data:
                    data[smell] = 0
            writer.writerow(data)
    
    print(f"\nAggregation complete. Results saved to {output_csv}")

if __name__ == '__main__':
    main()