import pandas as pd
import numpy as np
import uproot
import awkward as ak
import glob
import random


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


def flattenMET(calo, oMET, flat_params=(0.94, 0.066, 0.04, 500)):

    a, b, c, cutoff = flat_params

    rand_arr = np.random.rand(len(oMET))
    puppi_pt = oMET['puppi_MetNoMu']

    puppiflat = oMET[rand_arr[puppi_pt > 0] * (a - puppi_pt**b / cutoff**b) < c]
    caloflat = calo[calo["event"].isin(puppiflat["event"])]
    
    return caloflat, puppiflat

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


def readCaloData(data):
    
    print("Reading in calo tower data")
    calo_ak = uproot.concatenate(data, filter_name=["L1EmulCaloTower_ieta", "L1EmulCaloTower_iet", "L1EmulCaloTower_iphi"], library="ak")
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


def readPuppiData(data, subtractMu=True):
    
    if subtractMu == True:
        print("Reading in PUPPI MET pT and phi")
        puppi_ak = uproot.concatenate(data, filter_name = ["PuppiMET_pt", "PuppiMET_phi"], library="ak")
        puppi = ak.to_dataframe(puppi_ak)
        # Get x and y components of PUPPI MET
        print("Calculating x and y components of PUPPI MET")
        puppi_ptx, puppi_pty = np.cos(puppi.PuppiMET_phi) * puppi.PuppiMET_pt, np.sin(puppi.PuppiMET_phi) * puppi.PuppiMET_pt
    
        # Read muon pT data
        print("Reading in muon pT and phi")
        muons_ak = uproot.concatenate(data, filter_name=["Muon_phi", "Muon_pt", "Muon_isPFcand"], library="ak")
        muons = ak.to_dataframe(muons_ak)

        # Create df of total muon ptx and pty for every event
        print("Calculating muon ptx and pty for each event")
        pfcand_muons = muons[muons["Muon_isPFcand"]]

        # Calculate x and y components of pt
        pfcand_muons['muon_ptx'] = np.cos(pfcand_muons['Muon_phi']) * pfcand_muons['Muon_pt']
        pfcand_muons['muon_pty'] = np.sin(pfcand_muons['Muon_phi']) * pfcand_muons['Muon_pt']
        
        print("Calculating PUPPI MET no Mu and reformatting data")
        muon_ptx_tot, muon_pty_tot = pd.Series(pfcand_muons.groupby(level=0)['muon_ptx'].sum()), pd.Series(pfcand_muons.groupby(level=0)['muon_pty'].sum())     # Convert data in dict to pandas series for computation with puppi pt series
        puppi_ptx_noMu, puppi_pty_noMu = puppi_ptx + muon_ptx_tot, puppi_pty + muon_pty_tot           # add muon pt to the met for both x and y componets
        puppi_METNoMu = np.sqrt((puppi_ptx_noMu)**2 + (puppi_pty_noMu)**2)      # add x and y components of met_noMu in quadrature to get overall puppi_metNoMu
        puppi = pd.DataFrame(puppi_METNoMu, columns=["puppi_MetNoMu"], index=puppi_METNoMu.index)
        puppi.index.name = None
        puppi = puppi.reset_index(names="event", inplace=False)
    
    else:
        print("Reading in PUPPI MET pT")
        puppi_ak = uproot.concatenate(data, filter_name = ["PuppiMET_pt"], library="ak")
        puppi = ak.to_dataframe(puppi_ak)

        event_col = puppi.index
        puppi.insert(loc = 0, column="event", value=event_col)
        
    return puppi


def compNTT4(caloTowers):

    df = caloTowers[(caloTowers["ieta"]>= -4) & (caloTowers["ieta"] <= 4)]
    df = df.drop(["ieta", "iphi"], axis=1)
    df = df.groupby(["event"]).count()
    df = df.rename(columns={"iet": "ntt4"})
    df["ntt4"] /= 5
    df["ntt4"] = df["ntt4"].round()
    df["ntt4"] = df["ntt4"].clip(upper=32)
    return df


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
    caloTowers['caloMET'] = np.sqrt(caloTowers.ietx**2 + caloTowers.iety**2) /2               # calculate MET from metx and mety
    caloTowers = caloTowers.drop(['ietx', 'iety'], axis=1)                             # drop unrequired columns
    caloTowers = np.floor(caloTowers)                                                  # take floor to match firmware 
    
    return caloTowers.caloMET.astype("float64")


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