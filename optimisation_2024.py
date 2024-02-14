print("Importing packages")
import pandas as pd
import numpy as np
import uproot
import awkward as ak
import glob
from scipy.optimize import *
from scipy.special import expit
from scipy import stats
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import pickle
import time
import random
from matplotlib import pyplot as plt
import sys

CPUs = 30 #cpu_count()
threshold = 80
lowEff = 0.05

fileName, folder = sys.argv[0], sys.argv[1]


def calcL1MET(dataframe):
    
    """
    This function calculates the MET from the energy deposited on the trigger towers.
    """
    caloTowers = np.copy(dataframe)
   
    """ The second step is to convert the collider axes to radians and then calculate the x and y projections of energy deposits: """
    caloTowers[:,1] = caloTowers[:,1] * ((2*np.pi)/72)     # convert iphi into radians
    
    ietx = caloTowers[:,2] * np.cos(caloTowers[:,1].astype(float))    # calculate x component of energy for each tower
    iety = caloTowers[:,2] * np.sin(caloTowers[:,1].astype(float))    # calculate y component of energy for each tower
    
    caloTowers = np.c_[ caloTowers, ietx, iety ]
    caloTowers = pd.DataFrame(data=caloTowers[:, [3,4,5]], columns=["event", "ietx", "iety"])
    caloTowers = caloTowers.groupby(['event']).sum()
    caloTowers['caloMET'] = np.sqrt(caloTowers.ietx**2 + caloTowers.iety**2) / 2               # calculate MET from metx and mety
    caloTowers = caloTowers.drop(['ietx', 'iety'], axis=1)                             # drop unrequired columns
    caloTowers = np.floor(caloTowers)                                                  # take floor to match firmware 

    
    return caloTowers.caloMET.astype("float64")


def compNTT4(calo):

    df = calo[(calo["ieta"]>= -4) & (calo["ieta"] <= 4)]
    df = df.drop(["ieta", "iphi"], axis=1)
    df = df.groupby(["event"]).count()
    df = df.rename(columns={"iet": "ntt4"})
    df["ntt4"] /= 5
    df["ntt4"] = df["ntt4"].round()
    df["ntt4"] = df["ntt4"].clip(upper=32)
    return df



def readPuppiData(data, subtractMu):
    assert subtractMu == True or subtractMu == False
    
    if subtractMu == True:
        print("Reading in PUPPI MET pT and phi")
        batches = [batch for batch in uproot.iterate(data, filter_name = ["PuppiMET_pt", "PuppiMET_phi"], library = "ak")]
        dat_ak = ak.concatenate(batches)
        dat = ak.to_dataframe(dat_ak)
        # Get x and y components of PUPPI MET
        print("Calculating x and y components of PUPPI MET")
        puppi_ptx, puppi_pty = np.cos(dat.PuppiMET_phi) * dat.PuppiMET_pt, np.sin(dat.PuppiMET_phi) * dat.PuppiMET_pt
    
        # Read muon pT data
        print("Reading in muon pT and phi")
        batches = [batch for batch in uproot.iterate(data, filter_name = ["Muon_phi", "Muon_pt", "Muon_isPFcand"], library = "ak")]
        dat_ak = ak.concatenate(batches)
        dat = ak.to_dataframe(dat_ak)
        # Create df of total muon ptx and pty for every event
        muon_dat = {
            "muon_ptx" : [],
            "muon_pty" : []
        }
        print("Calculating muon ptx and pty for each event")
        for ev in set(dat.index.get_level_values(0)):
            df = dat.xs(ev, level='entry')                 # Get data for every muon in event ev
            df = df[df["Muon_isPFcand"]==True]
            muon_ptx = np.cos(df.Muon_phi) * df.Muon_pt    # Calculate x component of pt of every muon in ev
            muon_pty = np.sin(df.Muon_phi) * df.Muon_pt    # Calculate y component of pt of every muon in ev
            
            muon_dat["muon_ptx"].append(muon_ptx.sum())    # Calculate total x pt of muons in ev and append to dict
            muon_dat["muon_pty"].append(muon_pty.sum())    # Calculate total x pt of muons in ev and append to dict
        
        print("Calculating PUPPI MET no Mu and reformatting data")
        muon_ptx_tot, muon_pty_tot = pd.Series(muon_dat["muon_ptx"]), pd.Series(muon_dat["muon_pty"])     # Convert data in dict to pandas series for computation with puppi pt series
        puppi_ptx_noMu, puppi_pty_noMu = puppi_ptx + muon_ptx_tot, puppi_pty + muon_pty_tot           # add muon pt to the met for both x and y componets
        puppi_METNoMu = np.sqrt((puppi_ptx_noMu)**2 + (puppi_pty_noMu)**2)      # add x and y components of met_noMu in quadrature to get overall puppi_metNoMu
        oMET = pd.DataFrame(puppi_METNoMu, columns=["puppi_MetNoMu"], index=puppi_METNoMu.index)
        oMET.index.name = None
        oMET = oMET.reset_index(names="event", inplace=False)
    
    else:
        print("Reading in PUPPI MET pT")
        offline_batches = [batch for batch in uproot.iterate(data, filter_name = "PuppiMET_pt", library = "pd")]
        oMET = pd.concat(offline_batches)
        event_col = oMET.index
        oMET.insert(loc = 0, column="event", value=event_col)
        
    return oMET



def readCaloData(data):
    
    print("Reading in calo tower data")
    calo_batches = [batch for batch in uproot.iterate(data, filter_name = ["L1EmulCaloTower_ieta", "L1EmulCaloTower_iet", "L1EmulCaloTower_iphi"], library = "ak")]
    calo_ak = ak.concatenate(calo_batches)
    calo = ak.to_dataframe(calo_ak)
    """ For calo data, create new "event" column from the index, set new index as  "event, eta and phi values", store this df in new variable """
    calo['event'] = calo.index.get_level_values(0)
    calo = calo.reset_index(inplace=False)
    calo = calo.drop(["entry", "subentry"], axis=1)
    # Rename columns
    calo = calo.rename(columns={
        "L1EmulCaloTower_ieta": "ieta",
        "L1EmulCaloTower_iphi": "iphi",
        "L1EmulCaloTower_iet" : "iet",
    })
    # Reorder columns
    calo = calo[["ieta", "iphi", "iet", "event"]]
    
    return calo


def cutMET(cuts, calo, oMET):

    offlineMET_quantity = oMET.columns[-1]
    onlineCut, offlineCut = cuts
    print("Cutting on online MET")
    
    """ Perform online cut first """
    L1MET = pd.DataFrame(calcL1MET(calo))
    events_in_range = list(L1MET[L1MET["caloMET"] > onlineCut ].index)
    calo = calo[calo["event"].isin(events_in_range)]
    oMET = oMET[oMET["event"].isin(events_in_range)]
    print("Cutting on offline MET")
    
    """ Perform offline cut """
    events_in_range = list( (oMET[oMET[offlineMET_quantity] < offlineCut ].index) )
    calo = calo[calo["event"].isin(events_in_range)]
    oMET = oMET[oMET["event"].isin(events_in_range)]  

    return calo, oMET



def flattenMET(calo, oMET, uniformMax=150, binWidth=15, nBinsUniform=10):
    
    print ("Flattening MET distribution")
    if isinstance(oMET, pd.Series):
        oMET = pd.DataFrame(oMET)
        oMET = df.reset_index(drop=False).rename(columns={"index":"event"})
        
    offlineVar = oMET.columns[-1]
    n = int(len(oMET[oMET[offlineVar]>uniformMax]) - len(oMET[oMET[offlineVar]>uniformMax+binWidth]))*nBinsUniform  # Number of rows to select
    bins = np.linspace(20, uniformMax, num=20)
    oMET['binned'] = pd.cut(oMET[offlineVar], bins=bins, labels=False)
    selected_rows = []
    for i in range(n):
        bin_rows = oMET[oMET['binned'] == i%20]
        if len(bin_rows) > 0:
            random_row = bin_rows.sample(1)
            selected_rows.append(random_row)
            oMET = oMET.drop(random_row.index)
    selected_rows = pd.concat(selected_rows)
    remaining_rows = oMET[oMET[offlineVar] > uniformMax]
    selected_rows = pd.concat([selected_rows, remaining_rows])
    
    print ('N towers, offline MET before : ',len(calo))
    calo = calo[calo["event"].isin(selected_rows)]
    oMET = oMET[oMET["event"].isin(selected_rows)]
    print ('N towers, offline MET after : ',len(calo))
    
    return calo, oMET



def subsetEvents(calo, oMET, subset):
    print("Subsetting events")
    offlineMET_quantity = oMET.columns[-1]
    events = np.array(oMET.event)    # events is a list of all the event indices which survived the cut, so event indices 0 to 10,000, and the list is ~ 8318 long
    num_of_fit_events = int( np.ceil(len(oMET) * subset) )
    fit_event_indices = np.array( random.sample(range(len(events)), num_of_fit_events) ).astype(int)   # Randomly select num_of_fit_events in the range 0 - 8318, these will be the indices of the fit events. Then the rest of indices in this range are assigned for validation:
    valid_event_indices = np.array( list(set(range(len(oMET.event))) - set(fit_event_indices)) ).astype(int)     # indices between 0-8318 which can be used to index events list, to get the events which will be used in gs
    
    fit_events, valid_events = events[fit_event_indices], events[valid_event_indices]
    fit_calo_events, fit_pfMET_events = calo[calo["event"].isin(fit_events)], oMET[oMET["event"].isin(fit_events)][offlineMET_quantity].astype("float64")
    valid_calo_events, valid_pfMET_events = calo[calo["event"].isin(valid_events)], oMET[oMET["event"].isin(valid_events)][offlineMET_quantity].astype("float64")

    return (fit_calo_events, fit_pfMET_events), (valid_calo_events, valid_pfMET_events)



def prepareInputs(dir, subset=1, cuts=(0, 0), flatten=False, subtractMuon=True, prop=1):

    # Get a list of files
    random.seed(42)
    filesInDir = glob.glob(dir)
    fileNames = random.sample(
        filesInDir,
        int(np.ceil( prop * len(filesInDir)) )
    )
    data = ["{fileName}:{tree}".format(fileName = f, tree = "Events;1") for f in fileNames]

    # Read in offline and online data
    calo = readCaloData(data = data)
    oMET = readPuppiData(data = data, subtractMu = subtractMuon)
    
    # Cut MET online and offline based on cuts argument
    if (cuts[0] != 0) and (cuts[1] != 0):
        calo, oMET = cutMET(cuts = cuts, calo = calo, oMET = oMET)

    # Flatten MET distribution at low MET to increase emphasis on higher MET region
    if flatten==True:
        calo, oMET = flattenMET(calo = calo, oMET = oMET)

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

############################################################################################################

def readPuppiDataOld(files):
    offTree = "l1JetRecoTree/JetRecoTree"
    offlineDat = ["{fileName}:{tree}".format(fileName = f, tree = offTree) for f in files]

    offline_batches = [batch for batch in uproot.iterate(offlineDat, filter_name = "pfMetNoMu", library = "pd")]
    df_oMET = pd.concat(offline_batches)

    df_oMET = df_oMET.reset_index()
    df_oMET = df_oMET.rename(columns={"pfMetNoMu" : "puppi_MetNoMu", "index" : "event"})

    return df_oMET

def readCaloDataOld(files):
    caloTree = "l1CaloTowerEmuTree/L1CaloTowerTree"
    calofiles = ["{fileName}:{tree}".format(fileName = f, tree = caloTree) for f in files]

    calo_batches = [batch for batch in uproot.iterate(calofiles, filter_name = ["ieta", "iet", "iphi"], library = "ak")]
    df_calo_ak = ak.concatenate(calo_batches)
    df_calo = ak.to_dataframe(df_calo_ak)

    df_calo_met = df_calo
    df_calo_met['event'] = df_calo_met.index.get_level_values(0)
    df_calo_met = df_calo_met.reset_index(inplace=False)
    df_calo_met = df_calo_met.drop(["entry", "subentry"], axis=1)

    return df_calo_met


def prepareInputsOld(dir, subset=1, cuts=(0, 0), flatten=False, prop=1):

    random.seed(42)
    filesInDir = glob.glob(dir)
    fileNames = random.sample(
        filesInDir,
        int(np.ceil( prop * len(filesInDir)) )
    )

    print("Reading PUPPI MET")
    oMET = readPuppiDataOld(fileNames)
    print("Reading calo data")
    calo = readCaloDataOld(fileNames)
    
    # Cut MET online and offline based on cuts argument
    if (cuts[0] != 0) and (cuts[1] != 0):
        calo, oMET = cutMET(cuts = cuts, calo = calo, oMET = oMET)

    # Flatten MET distribution at low MET to increase emphasis on higher MET region
    if flatten==True:
        calo, oMET = flattenMET(calo = calo, oMET = oMET)

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


############################################################################################

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

#################################################################################

print("Loading data")
data = folder + "L1Ntuple_*.root"
fit, valid = prepareInputsOld(dir = data, subset=0.7, cuts=(0, 250))
calo, puppi, ntt4 = fit


def objective(params, turn_on = True):
    a, b, c, d = params
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
        difference = MET - puppi
        sqrd_diff = np.sum(difference**2)
        print("Squared difference = {}".format(np.round(sqrd_diff,2)))
        return sqrd_diff


print("Starting optimisation")
bounds = [(0, 4), (0, 3), (0, 4), (0, 4)]
result = differential_evolution(objective, bounds, popsize=40, maxiter=100, strategy="best1bin")

print(result.x)
print(result)