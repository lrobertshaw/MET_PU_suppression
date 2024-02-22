print("Importing packages")
import numpy as np
from scipy.optimize import differential_evolution
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import sys

from packages.functions import *

global CPUs
CPUs = 30 #cpu_count()


def prepareInputs(dir, subset=1, cuts=(0, 0), flatten=(0.94, 0.066, 0.04, 500), subtractMuon=True, oldData=False):

    if oldData == False: # Read in offline and online data
        data = dir + ":Events;1"
        calo = readCaloData(data = data)
        oMET = readPuppiData(data = data, subtractMu = subtractMuon)
    else:
        data = dir
        calo = readCaloDataOld(data = data)
        oMET = readPuppiDataOld(data = data)

    # Cut MET online and offline based on cuts argument
    if (cuts[0] != 0) and (cuts[1] != 0):
        calo, oMET = cutMET(cuts = cuts, calo = calo, oMET = oMET)

    # Flatten MET distribution at low MET to increase emphasis on higher MET region
    if flatten != False:
        calo, oMET = flattenMET(calo = calo, oMET = oMET, flat_params = flatten)

    # Subset events into fitting and testing samples
    fit_events, valid_events = subsetEvents(calo=calo, oMET=oMET, subset=subset)
    
    # Calculate NTT4
    print("Calculating NTT4")
    fit_calo_events, valid_calo_events = fit_events[0], valid_events[0]
    compntt_fit, compntt_valid = compNTT4(fit_calo_events), compNTT4(valid_calo_events)

    # Package and return data
    fit_data = (fit_events[0], fit_events[1], compntt_fit)
    valid_data = (valid_events[0], valid_events[1], compntt_valid)
    
    return fit_data, valid_data


def applyCaloTowerThresh(caloTowers, a, b, c, d):
    
    caloTowersPUsup = caloTowers.copy()
    compntt = compNTT4(caloTowersPUsup)
    MET = np.empty((0, 4))
    ietas = np.unique(caloTowersPUsup["ieta"][caloTowersPUsup["ieta"] != 0])    # Create unique list of all eta
    
    def process_ieta(ieta):
        towers = caloTowersPUsup[caloTowersPUsup["ieta"] == ieta].to_numpy()    # Convert pandas dataframe to numpy array
        thresholds = towerEtThreshold(ieta, compntt, a, b, c, d)    # Does compntt need to be calculated in the function so calculated for every thread?
        passed_tows = applyThresholds(towers[:, [2,3]], thresholds )
        towers[:,2] = towers[:,2].astype(float) * passed_tows
        return towers

    with ThreadPool(CPUs) as pool:
        towers_list = pool.map(process_ieta, ietas)
    
    MET = np.concatenate(towers_list)
    
    return calcL1MET(MET), MET.astype(int)


def objective(params, *args):
    a, b, c, d = params
    calo, puppi, _ = args[0]
    turn_on, threshold, lowEff = args[1]
    print("\nCurrently trying: a = {}, b = {}, c = {} and d = {}".format(np.round(a,2), np.round(b,2), np.round(c,2), np.round(d,2)))
    MET, _ = applyCaloTowerThresh(calo, a, b, c, d)
    
    if turn_on == True:
        offline_bins = np.linspace(0, 300, 60)
        eff_0p05 = 0
        foundEff0p05 = False
        eff_0p95 = 99999
        eff_before = 0
        x_cross_95 = 99999
        x_cross_05 = 0
        eff_0p5 = 0
        foundEff0p5 = False
        for i in range(len(offline_bins) - 1):
            offline_range = (puppi >= offline_bins[i]) & (puppi < offline_bins[i + 1])
            num_offline = sum(offline_range)
            num_both = sum((MET > threshold) & offline_range)
            if num_offline > 0:
                eff = num_both / num_offline
            else:
                eff = 0
            # print (i,offline_bins[i],eff,num_offline,num_both)
            if eff >= lowEff and not foundEff0p05 :
                eff_0p05 = offline_bins[i]
                if (i>0):
                    x_cross_05 = offline_bins[i-1] + ((lowEff - eff_before) / (eff - eff_before)) * (eff_0p05 - offline_bins[i-1])
                else : x_cross_05 = eff_0p05
                foundEff0p05 = True
    
            if eff >= 0.5 and not foundEff0p5:
                eff_0p5 = offline_bins[i]
                foundEff0p5 = True
            if eff >= 0.95 :
                eff_0p95 = offline_bins[i]
                x_cross_95 = offline_bins[i-1] + ((0.95 - eff_before) / (eff - eff_before)) * (eff_0p95 - offline_bins[i-1])
                break
    
            eff_before = eff
        #print (a, b, c, d, eff_0p05, eff_0p5, eff_0p95, eff_0p95-eff_0p05, x_cross_05, x_cross_95, x_cross_95-x_cross_05)
        # return (eff_0p95-eff_0p05)
        print("Turn on width: {}".format(np.round(x_cross_95 - x_cross_05, 2)))
        return(x_cross_95-x_cross_05)
    
    else:
        error = MET - puppi
        rmse = np.sqrt(np.mean(error**2))
        print("RMSE = {}".format(np.round(rmse,2)))
        return rmse


if __name__ == "__main__":

    print("Parse args")
    
    data = sys.argv[1]
    
    if sys.argv[2] == "turnon":
        turn_on_options = (True, 80, 0.05)
    elif sys.argv[2] == "rmse":
        turn_on_options = (False, 0, 0)
    else:
        raise Exception("choose turnon or rmse")
    
    workers = sys.argv[3]

    print("Loading data")
    fit, valid = prepareInputs(dir = data, subset=0.7, cuts=(0, 250))

    print("Starting optimisation")
    bounds = [(0, 4), (0, 3), (0, 4), (0, 4)]
    x0 = (2.0, 2.0, 0.5, 2.0)

    result = differential_evolution(
        func     = objective,
        bounds   = bounds,
        args     = (fit, turn_on_options),
        
        x0 = x0,
        popsize  = 15,    # 15
        maxiter  = 1000,    # 1000
        strategy = "best1bin",    # "best1bin"
        init     = "sobol",    #"latinhypercube"
        disp     = True,
        workers  = workers,    # 1
        polish   = False
        )

    print(result.x)
    print(result)