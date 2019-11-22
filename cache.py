import numpy as np
import math

class Cache():
    def __init__(self, max_size=10):
        self.cache = []
        self.size = 0
        self.max_size=max_size
    
    def add(self, element):
        self.cache.append(element)
        self.size+=1
        if self.size > self.max_size:
            del self.cache[0]
            self.size = self.max_size
    
    def mean(self, i):
        column = [element[i] for element in self.cache]
        return np.mean(np.array(column), axis=0)

    def empty(self):
        return self.size == 0

    def get_size(self):
        return self.size

    def get_last(self):
        return self.cache[self.size-1]

    def get_all(self):
        return self.cache
    
    def get_all_index(self, i):
        return [row[i] for row in self.cache]

    def print_cache(self):
        for e in self.cache:
            print(e)

    
if __name__ == '__main__':
    print('===Test Cache===')
    cache = Cache(max_size=5)
    cache.add([5,4])
    print(cache.get_size())
    print(cache.print_cache())

    cache.add([8,1])
    cache.add([3,2])
    cache.add([4,5])
    cache.add([6,2])
    print(cache.get_size())
    print(cache.print_cache())

    cache.add([1,4])
    print(cache.get_size())
    print(cache.print_cache())
    print(cache.mean())
