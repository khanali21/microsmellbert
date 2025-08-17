import os
import shutil

# Define paths
repo_root = '../spring-petclinic-microservices'
output_root = './data/services'
services = [
    'spring-petclinic-api-gateway',
    'spring-petclinic-customers-service',
    'spring-petclinic-vets-service',
    'spring-petclinic-visits-service',
    'spring-petclinic-admin-server',
    'spring-petclinic-config-server',  # Shared config service
    'spring-petclinic-discovery-server'  # Shared discovery service
]

# Create output dirs and copy .java files
for service in services:
    src_dir = os.path.join(repo_root, service, 'src/main/java')
    dest_dir = os.path.join(output_root, service)
    os.makedirs(dest_dir, exist_ok=True)
    
    if os.path.exists(src_dir):
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.java'):
                    shutil.copy(os.path.join(root, file), dest_dir)
    else:
        print(f"Warning: {src_dir} not found for {service}")

print("Java files extracted.")