import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import numpy as np

def main():
    print("Starting embeddings extraction...")  # Debug start
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")  # Debug device (CPU on Mac)

    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = RobertaModel.from_pretrained('microsoft/codebert-base').to(device)
    print("CodeBERT loaded.")  # Debug model load

    data_path = 'data/services'
    embeddings = []
    for service in os.listdir(data_path):
        print(f"Processing service: {service}")  # Debug per service
        service_dir = os.path.join(data_path, service)
        service_code = ''
        pre_files = [f for f in os.listdir(service_dir) if f.endswith('.pre')]
        if not pre_files:
            print(f"No preprocessed files for {service}; skipping.")
            continue
        for file in pre_files:
            with open(os.path.join(service_dir, file), 'r') as f:
                service_code += f.read() + '\n'
            print(f"  Added file: {file}, code length: {len(service_code)}")  # Debug code buildup
        
        if len(service_code.strip()) == 0:
            print(f"Empty code for {service}; skipping.")
            continue
        
        try:
            inputs = tokenizer(service_code, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            emb = outputs.pooler_output.cpu().detach().numpy()  # 768D vector
            print(f"  Embedding shape: {emb.shape}")  # Debug emb
            embeddings.append({'service': service, 'embedding': emb.flatten()})
        except Exception as e:
            print(f"Embedding error for {service}: {e}")

    if embeddings:
        df_emb = pd.DataFrame(embeddings)
        df_emb.to_pickle('embeddings/embeddings.pkl')  # Pickle for numpy arrays
        print("embeddings.pkl saved. Num services:", len(df_emb))  # Debug save
    else:
        print("No embeddings extracted; check preprocessing.")

    print("Embeddings extraction complete.")  # Debug end

if __name__ == "__main__":
    main()