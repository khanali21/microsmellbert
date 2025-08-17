import os
import pandas as pd

# Heuristics (adapt from your chapter/train-ticket)
def is_mega_service(num_classes, loc_mean):
    return num_classes > 15 or loc_mean > 50  # Example thresholds

def is_crudy_service(operations):  # Need to parse for CRUD ops (e.g., count GET/POST/PUT/DELETE in controllers)
    crud_count = operations.count('create') + operations.count('read') 
    return crud_count > 0.7 * len(operations)

def is_ambiguous_service(responsibilities):
    return len(responsibilities) > 3  # Example: too many unrelated funcs

# Collect data per service (simplified; parse .java for real metrics)
services_dir = 'data/services'
labels = []
for service in os.listdir(services_dir):
    service_path = os.path.join(services_dir, service)
    if os.path.isdir(service_path):
        num_classes = len([f for f in os.listdir(service_path) if f.endswith('.java')])
        loc_mean = sum(len(open(os.path.join(service_path, f)).readlines()) for f in os.listdir(service_path) if f.endswith('.java')) / num_classes
        # Pseudo: extract ops/responsibilities (use javalang to parse methods)
        operations = []  # TODO: Parse @GetMapping, etc.
        responsibilities = []  # TODO: Semantic analysis
        
        mega = 1 if is_mega_service(num_classes, loc_mean) else 0
        crudy = 1 if is_crudy_service(operations) else 0
        ambiguous = 1 if is_ambiguous_service(responsibilities) else 0
        
        labels.append([service, mega, crudy, ambiguous])

# Save to CSV
df = pd.DataFrame(labels, columns=['Service', 'Mega-Service', 'CRUDy Service', 'Ambiguous Service'])
df.to_csv('data/labels_petclinic.csv', index=False)
print("Labels generated: data/labels_petclinic.csv")