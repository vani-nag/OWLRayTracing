import numpy as np
import pandas as pd

coords = np.random.rand(400000, 2)*2
coords = coords.round(3)
#print(coords)
coords = pd.DataFrame(coords)
coords.to_csv("input.csv",index=None,columns=None)
