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
        calo = readCaloDataOld(files = data)
        oMET = readPuppiDataOld(files = data)

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
    
    res = calcTurnOn(MET=MET, puppi=puppi, threshold=threshold, lowEff=lowEff) if turn_on == True else calcRMSE(MET=MET, puppi=puppi)
    print("{var}: {val}".format(var = "Width" if turn_on==True else "RMSE", val = res))
    
    return res


if __name__ == "__main__":

    print("Parsing args")
    
    data = sys.argv[1]
    
    if sys.argv[2] == "turnon":
        turn_on_options = (True, 80, 0.10)
    elif sys.argv[2] == "rmse":
        turn_on_options = (False, 0, 0)
    else:
        raise Exception("choose turnon or rmse")
    
    workers = sys.argv[3]

    print("Loading data")
    fit, _ = prepareInputs(dir = data, subset=1.0, cuts=(0, 300))

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
