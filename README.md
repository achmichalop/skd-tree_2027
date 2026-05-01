# In-memory Multidimensional Indexing Using the skd-tree
This repository contains the code and resources used to reproduce the experiments presented in the paper. The source code for the **skd-tree** index, which is the main method introduced in the paper, is located at `/indices/nonlearned/skdtree`.

## List of Competitor Indices
This repository includes implementations of all indices that are compared against **skd-tree** in the experimental evaluation of the paper:
- **kd-tree**
- **R-tree**
- **PH-tree**
- **Adaptive Grid (AG)**
- **Uniform Grid (UG)**
- **Flood**
- **IFI**

The **kd-tree**, **R-tree**, **Uniform Grid (UG)**, **Flood** and **IFI** indices are adapted from the [learnedbench repository](https://github.com/qyliu-hkust/learnedbench).

The **PH-tree** implementation is adapted from the [phtree-cpp repository](https://github.com/tzaeschke/phtree-cpp).

The **Adaptive Grid (AG)** method is our own implementation, developed based on the **Uniform Grid (UG)** approach and using the splitting strategy of the **skd-tree**.

## Dependencies
This project requires the following dependencies to build and run the experiments. Some are required for all indices, while others are specific to the **skd-tree** index.

### General Dependencies (for all indices)
- **Boost 1.79.0** - [Boost 1.79.0](https://www.boost.org/users/history/version_1_79_0.html)

### Dependencies for skd-tree
- **g++ / gcc** – version 13 or higher
-  **AVX-512** – CPU must support AVX-512 instructions
- **Armadillo** – [Armadillo](https://arma.sourceforge.net/)
- **ensmallen** – [ensmallen](https://github.com/mlpack/ensmallen)

## Build Benchmark
All indices in this repository are **header-only** (`.hpp`) files.  
As a result, **no individual build is required** for the indices. You can include them directly in your project or in a single benchmark executable.

### Modifications
Before building the benchmark, you need to ensure that CMake can find the correct Boost library version.
1. Open the `CMakeLists.txt` file in the root of the repository.
2. Set the following variables to the correct path of your Boost 1.79.0 installation:

```cmake
set(BOOST_ROOT "PATH TO boost_1_79_0")
set(Boost_INCLUDE_DIR "PATH TO boost_1_79_0")
set(Boost_LIBRARY_DIR "PATH TO boost_1_79_0/stage/lib")
```

### Compilation
Once all dependencies are installed and `CMakeLists.txt` is configured, you can build the benchmark with the following commands:

```bash
# Create a build directory and navigate into it
mkdir build && cd build

# Generate the build files with CMake
cmake ..

# Compile the project
make
```
## Run Experiments
### 1. Prepare Data
We provide a script that generates uniform datasets with 20 million points for different dimensionalities (d = 2, 4, 6, 8) Both the cardinality and dimensionality of datasets can be changed. 
```bash
cd data
python3 create_data.py
```
The script generates the following folders:
- **datasets/** - containing the initial datasets
- **range_queries/** and **knn_queries/** - each containing 20 queries for every dataset

### 2. Run Queries
Once the datasets and benchmark executable are ready, you can run queries for each index.

#### 2.1 Range and kNN Queries
For **range queries** the selectivity is set to 0.001%. For **knn queries** each point is evaluated for different k values (k=1,5,10,50,100,500).

The general command format is:

```bash
# General command format
./bin/bench_<n>d <index> <dataset_fname> <num_points> <queries_fname> <num_queries> range
./bin/bench_<n>d <index> <dataset_fname> <num_points> <queries_fname> <num_queries> knn
```

Example usage:

```bash
# Run range queries for a 4-dimensional dataset
./bin/bench_4d skdtree ../data/datasets/20M_4D_uniform.bin 20000000 ../data/range_queries/20M_4D_uniform_range_queries.txt 20 range

# Run kNN queries for a 4-dimensional dataset
./bin/bench_4d skdtree ../data/datasets/20M_4D_uniform.bin 20000000 ../data/knn_queries/20M_4D_uniform_knn_queries.txt 20 knn
```
##### Notes:
- Replace `<n>` with the dataset dimensionality
- Replace `<index>` with the index you want to test (e.g., skdtree)
- Adjust `<num_points>` and `<num_queries>` if using custom datasets/queries.

#### 2.2 Mixed Workload
The benchmark also supports a **mixed workload**, which combines dynamic updates (**insertions** and **deletions**) with an equal number of **range** and **knn queries**, as described in the paper.

The general command format is:
```bash
./bin/bench_<n>d_mixed <index> <dataset_fname> <num_points> <update_ratio> <range_queries> <knn_queries> <num_queries> 
```
Example usage:
```bash
# Run a mixed workload for a 4-dimensional dataset
./bin/bench_4d_mixed skdtree ../data/datasets/20M_4D_uniform.bin 20000000 0.1 ../data/range_queries/20M_4D_uniform_range_queries.txt ../data/knn_queries/20M_4D_uniform_knn_queries.txt 20 
```



