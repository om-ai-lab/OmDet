import pickle as pkl
import lmdb
from collections import OrderedDict


class LRUCache:
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def has(self, key) -> bool:
        return key in self.cache

    # we return the value of the key
    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    # first, we add / update the key by conventional methods.
    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key, value) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def pop(self, key, value):
        self.cache.pop(key, None)


class LmdbReader:
    def __init__(self, path):
        self.path = path
        self.env = self.init_lmdb(path)

    def init_lmdb(self, l_path):
        env = lmdb.open(
            l_path, readonly=True,
            create=False, lock=False)  # readahead=not _check_distributed()
        txn = env.begin(buffers=True)
        return txn

    def read(self, _id):
        try:
            value = self.env.get(str(_id).encode("utf-8"))
            value = pkl.loads(value)
            return value
        except Exception as e:
            print("Error in reading {} from {}".format(_id, self.path))
            raise e
