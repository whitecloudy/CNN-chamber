import os
import pickle
from pathlib import Path

def make_cache_hashname(datas, header=None, tail=None):
    import hashlib
    tuples_byte = pickle.dumps(datas)
    hash_handler = hashlib.sha3_224()
    hash_handler.update(tuples_byte)

    cache_filename = hash_handler.digest().hex()

    if header is not None:
        cache_filename = header + cache_filename

    if tail is not None:
        cache_filename = cache_filename + tail

    return cache_filename


def save_cache(save_data, cache_filename):
    import os
    from pathlib import Path
    cache_path = str(Path.home()) + "/data/cache/" + cache_filename

    with open(cache_path, "wb") as cache_file:
        pickle.dump(save_data, cache_file)


def load_cache(cache_filename, testing=False):
    if testing:
        cache_path = cache_filename
    else:
        cache_path = str(Path.home()) + "/data/cache/" + cache_filename

    rt_data = None
    print(cache_path)
    if os.path.isfile(cache_path):
        print("Cache file found")
        with open(cache_path, "rb") as cache_file:
            rt_data = pickle.load(cache_file)
    else:
        print("No Cache file found")

    return rt_data

