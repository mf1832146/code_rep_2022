

import numpy as np

a = np.ones((3,3))
b = np.zeros((4,4))
#b[:3, :3] = a
#print(b[:3, :3])

b[:3, :3] = a[1]
print(b)