import time
import random
# __all__ = ['generateId']

def generateId():
    return str(time.time()) + str(random.randint(1, 1000000))