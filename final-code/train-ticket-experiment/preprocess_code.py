import os
import re
import javalang  # For better parsing if needed

def preprocess_code(file_path):
    with open(file_path, 'r') as f:
        code = f.read()
    
    # Remove comments (regex for single/multi-line)
    code = re.sub(r'//.*|/\*[\s\S]*?\*/', '', code)
    
    # Normalize variables/identifiers (replace with 'VAR'; basic regex)
    code = re.sub(r'\b[A-Za-z_]\w*\b', 'VAR', code)
    
    # Exclude auto-generated: Simple check for '@Generated' annotation
    if '@Generated' in code:
        return None  # Skip file
    
    return code

# Apply to all extracted files
data_path = 'data/services'
for service in os.listdir(data_path):
    service_dir = os.path.join(data_path, service)
    for file in os.listdir(service_dir):
        if file.endswith('.java'):
            file_path = os.path.join(service_dir, file)
            preprocessed = preprocess_code(file_path)
            if preprocessed:  # Only save if not skipped
                with open(file_path + '.pre', 'w') as f:
                    f.write(preprocessed)

print("Preprocessing complete.")