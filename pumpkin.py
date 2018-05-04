import math
import thinkstats

def Pumpkin(items):
    mean, var = thinkstats.MeanVar(items)
    stdev = math.sqrt(var)
    return mean, var, stdev

