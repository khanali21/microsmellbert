# quick one-off in a Python shell
import pandas as pd, numpy as np, pickle

with open("embeddings/embeddings_spring_petclinic.pkl","rb") as f:
    obj = pickle.load(f)

if isinstance(obj, pd.DataFrame):
    # drop non-numeric cols like 'Service'
    arr = obj.select_dtypes(include=[np.number]).values.astype(np.float32)
elif isinstance(obj, dict):
    # if dict[id]->vec and your labels have no id, this won't align—prefer Option A.
    # if you do have ids, build the array in that order.
    raise SystemExit("Dict embeddings: prefer Option A (align by id).")
else:
    # try to coerce (e.g., list/ndarray)
    arr = np.asarray(obj).astype(np.float32)

np.save("embeddings/embeddings_spring_petclinic_clean.npy", arr)
print(arr.shape)
