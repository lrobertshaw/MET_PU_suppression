import numpy as np

def getTurnOn( online, offline, threshold=80 ) :
    offline_bins = np.linspace(0, 300, 31)
    efficiency = []

    for i in range(len(offline_bins) - 1):
        # Define the offline range for this bin
        offline_range = (offline >= offline_bins[i]) & (offline < offline_bins[i + 1])
        # count the number of events passing the threshold in the offline range
        num_offline = sum(offline_range)
        # count the number of events passing the threshold in both online and offline ranges
        num_both = sum((online > threshold) & offline_range)
        # calculate the efficiency as the ratio of online events passing the cut over offline events passing the threshold
        if num_offline > 0:
            eff = num_both / num_offline
        else:
            eff = 0
        efficiency.append(eff)

    bin_centers = (offline_bins[:-1] + offline_bins[1:]) / 2

    return bin_centers, efficiency


def threshold_calc(ieta, ntt4, a, b, c, d, scale=False):
    towerAreas = [    0., # dummy for ieta=0
                  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                  1.03,1.15,1.3,1.48,1.72,2.05,1.72,4.02,
                  0., # dummy for ieta=29
                  3.29,2.01,2.02,2.01,2.02,2.0,2.03,1.99,2.02,2.04,2.00,3.47]
    
    numerator = (towerAreas[int(abs(ieta))]**a) * (ntt4**c)
    denominator = d * (1 + np.exp(-b * (abs(ieta))))
    
    threshold = (numerator / denominator).clip(max=40)
#    return (threshold/2)# / towerAreas[int(abs(ieta))]
    if towerAreas[int(abs(ieta))] == 0:
        return 0
    else:
        return (threshold/2)# / towerAreas[int(abs(ieta))]


def lookup_gen(params, splitEta=False):
    
    all_pu_bins = np.linspace(0, 31, 32)

    if splitEta==True:
        barrel_ieta, endcap_ieta, forward_ieta = np.linspace(0, 16, 17), np.linspace(17, 28, 12), np.linspace(29, 41, 13)
        res = []
        for ieta in barrel_ieta:
            for pu_bin in all_pu_bins:
                thresh = threshold_calc(ieta, pu_bin, *params[:4])
                res.append((ieta, pu_bin, thresh))
        for ieta in endcap_ieta:
            for pu_bin in all_pu_bins:
                thresh = threshold_calc(ieta, pu_bin, *params[4:8])
                res.append((ieta, pu_bin, thresh))
        for ieta in forward_ieta:
            for pu_bin in all_pu_bins:
                thresh = threshold_calc(ieta, pu_bin, *params[8:])
                res.append((ieta, pu_bin, thresh))
        return res
    
    else:
        all_ieta_vals = np.linspace(0, 41, 42)
        res = []
        for ieta in all_ieta_vals:
            for pu_bin in all_pu_bins:
                thresh = threshold_calc(ieta, pu_bin, *params)
                res.append((ieta, pu_bin, thresh))
        return res


def getResidual( online, offline ) :
    offline_bins = np.linspace(0, 300, 31)

    q68s = []
    q95s = []
    responses = []
    resolutions = []
    for i in range(len(offline_bins) - 1):
        # Define the offline range for this bin
        offline_range = (offline >= offline_bins[i]) & (offline < offline_bins[i + 1]) #& (online < 1000)
        offline_inBin = offline[offline_range]
        online_inBin = online[offline_range]
        res = (offline_inBin-online_inBin)
        # residual.append(res)
        q68 = np.percentile(res, [16, 84])
        q95 = np.percentile(res, [2.5, 97.5])
        # print (offline_bins[i], offline_bins[i+1], np.abs(q68[0]-q68[1] ))
        q68s.append( np.abs(q68[0]-q68[1] ) )
        q95s.append( np.abs(q95[0]-q95[1] ) )

        responses.append( np.mean(online_inBin/offline_inBin) )
        resolutions.append( np.mean((online_inBin-offline_inBin)/offline_inBin) )
        # print (online_range)
    # print (efficiency)
    bin_centers = (offline_bins[:-1] + offline_bins[1:]) / 2

    return bin_centers, q68s, q95s, responses, resolutions