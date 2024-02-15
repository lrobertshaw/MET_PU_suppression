import numpy as np


def towerEtThreshold(ieta, ntt4, a, b, c, d):
    
    pu = ntt4.copy()
    towerAreas = [    0., # dummy for ieta=0
                  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                  1.03,1.15,1.3,1.48,1.72,2.05,1.72,4.02,
                  0., # dummy for ieta=29
                  3.29,2.01,2.02,2.01,2.02,2.0,2.03,1.99,2.02,2.04,2.00,3.47]
    
    term1 = float(towerAreas[int(abs(ieta))] ** a)
    term2 = 1 / ( d * (1 + np.exp(-b * (abs(ieta)))) )
    term3 = pu["ntt4"] ** c
    
    pu.rename(columns={"ntt4": "threshold"})
    
    pu["threshold"] = (term1 * term2 * term3).clip(upper=40)    # Rounding makes big difference to low Et towers as 0.6 --> 1.0 thus 0.5GeV towers no longer pass.
    
    return pu["threshold"].to_dict()    # Returns a vector - a threshold for every event because there's a compNTT4 value for every event


def applyThresholds(data, thresholds):  
    iets = np.array(data[:,0])
    thresholds = np.array([thresholds[event] for event in data[:,1]])
    comparison_list = iets > thresholds
    return comparison_list


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

    num_processes = CPUs #cpu_count()
    with ThreadPool(num_processes) as pool:
        towers_list = pool.map(process_ieta, ietas)
    
    MET = np.concatenate(towers_list)
    
    return calcL1MET(MET), MET.astype(int)