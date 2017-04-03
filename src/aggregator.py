import numpy as np

def basic_aggregator(data):
    return [min(data), max(data), np.mean(data), np.std(data), np.median(data)]
