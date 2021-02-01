import numpy as np
list = [1,2,0,4,5]
index = np.argmin(np.array(list))

list.pop(index)
print(list)