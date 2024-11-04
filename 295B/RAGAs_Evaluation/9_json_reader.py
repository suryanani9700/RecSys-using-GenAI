import os

# Read and print contents of the .deepeval-cache.json file
with open(os.path.expanduser(".deepeval-cache.json"), "r") as cache_file:
    print(cache_file.read())
