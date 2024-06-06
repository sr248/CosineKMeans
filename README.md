# CosineKMeans
Expand scikit-learn KMeans clustering to calculate with cosine distance instead of Euclidean distance.


It can be used in same way as the original scikit-learn KMeans.

## Easy way to get start
1. Setup the Cython files.
```bash
% python setup.py build_ext --inplace  
```
  
2. Run CosineKMeans in Python.
```python
from cosine_kmeans import CosineKMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
ckm = CosineKMeans(n_clusters=2, n_init=5).fit(X)
ckm.labels_
ckm.predict([[0, 0], [12, 3]])
ckm.cluster_centers_
```