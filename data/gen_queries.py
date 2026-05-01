import sys
import os
import random
from array import array

def read_dataset(file_path, n, d):
    with open(file_path, "rb") as f:
        total_values = n * d
        data = array('d')
        data.fromfile(f, total_values)
    points = [data[i*d:(i+1)*d] for i in range(n)]
    return points



dataset_folder = os.path.abspath(sys.argv[1])
parent_folder = os.path.dirname(dataset_folder)
range_folder = os.path.join(parent_folder, "range_queries")
knn_folder = os.path.join(parent_folder, "knn_queries")

os.makedirs(range_folder, exist_ok=True)
os.makedirs(knn_folder, exist_ok=True)

selectivity = 0.00001


for file in os.listdir(dataset_folder):
    dataset_path = os.path.join(dataset_folder, file)

    info = file.split('_')
    size = int(info[0][:-1]) * 1000000
    dim = int(info[1][:-1])

    points = read_dataset(dataset_path, size, dim)

    mins = [float('inf')] * dim
    maxs = [float('-inf')] * dim
    for p in points:
        for j in range(dim):
            if p[j] < mins[j]:
                mins[j] = p[j]
            if p[j] > maxs[j]:
                maxs[j] = p[j]

    ranges = [(maxs[j] - mins[j]) if maxs[j] > mins[j] else 1.0 for j in range(dim)]

    normalized_points = []
    for p in points:
        norm_p = [(p[j] - mins[j]) / ranges[j] for j in range(dim)]
        normalized_points.append(norm_p)

    sample_points = random.sample(normalized_points, 20)

    knn_file = os.path.join(knn_folder, file.replace(".bin", "_knn_queries.txt"))
    
    with open(knn_file, "w") as f:
        for p in sample_points:
            line = ",".join(str(round(coord, 6)) for coord in p)
            f.write(line + "\n")
    

    side_length = selectivity ** (1.0 / dim)
    range_file = os.path.join(range_folder, file.replace(".bin", "_range_queries.txt"))

    with open(range_file, "w") as f:
        for p in sample_points:
            low = p
            high = [min(1.0, coord + side_length) for coord in low]
            low_str = ",".join(str(round(coord, 6)) for coord in low)
            high_str = ",".join(str(round(coord, 6)) for coord in high)
            f.write(f"{low_str} {high_str}\n")
