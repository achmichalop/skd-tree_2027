import sys
import random
import os
from array import array


dimensions = [2,4,6,8]
size = int(sys.argv[1])
folder_path = sys.argv[2]
n_millions = size // 1000000

os.makedirs(folder_path, exist_ok=True)

for d in dimensions:
    filename = os.path.join(folder_path,  str(n_millions) + "M_" + str(d) + "D_uniform.bin")
    values = size * d
    data = array('d', (random.random() for _ in range(values)))

    with open(filename, "wb") as f:
        data.tofile(f)

