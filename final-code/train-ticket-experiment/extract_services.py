import os
import shutil

repo_path = 'train-ticket'
data_path = 'data/services'
os.makedirs(data_path, exist_ok=True)

for service_dir in os.listdir(repo_path):
    if service_dir.startswith('ts-'):
        service_path = os.path.join(repo_path, service_dir, 'src/main/java')
        if os.path.exists(service_path):
            target_dir = os.path.join(data_path, service_dir)
            os.makedirs(target_dir, exist_ok=True)
            for root, _, files in os.walk(service_path):
                for file in files:
                    if file.endswith('.java'):
                        shutil.copy(os.path.join(root, file), target_dir)