import subprocess
import sys


size = 20000000        
dataset_folder = "datasets"

print("Generate uniform data inside the folder", dataset_folder)
subprocess.run(["python3", "gen_data.py", str(size), dataset_folder], check=True)

print("Generate queries for each dataset")
subprocess.run(["python3", "gen_queries.py", dataset_folder], check=True)

print("Finished!")