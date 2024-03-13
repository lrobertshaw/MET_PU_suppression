import numpy as np
import awkward as ak

def efficiency(data_on, data_off, threshold, binwidth, xmax):
    
    num = ak.zip({'on': data_on})
    num['off'] = data_off
    num = num[num['on'] > threshold]

    numHist = np.histogram(num['off'], bins=int(xmax/binwidth), range=(0,xmax))[0]
    denomHist  = np.histogram(data_off, bins=int(xmax/binwidth), range=(0,xmax))[0]
    effs = numHist/denomHist
    xvals = [x+(binwidth/2) for x in range(0,xmax,binwidth)]

    return effs, xvals

def getThreshForRate(rates, bins, target_rate):
    
    finalThresh = 0
    for rate, thresh in zip(rates, range(bins)):
            if rate < target_rate:
                finalThresh = thresh
                break
    return finalThresh