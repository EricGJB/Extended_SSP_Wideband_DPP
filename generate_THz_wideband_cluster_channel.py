from scipy import io

import numpy as np
random_seed = 2022
np.random.seed(random_seed)

# fixed system parameters
fc, W, tau_max, num_subpaths = 100 * 1e9, 10 * 1e9, 20 * 1e-9, 10

# varaibles 
num_antenna_bs, num_antenna_ue, num_sc, num_samples, num_clusters = 256, 4, 128, 200, 4

AS = 5
b_angle = np.sqrt(AS**2/2) # the variance of Laplace distribution is 2b^2
DS = 1*1e-9
b_delay = np.sqrt(DS**2/2)

eta = W / num_sc
Lp = num_clusters * num_subpaths

H_list = np.zeros((num_samples, num_sc, num_antenna_ue, num_antenna_bs),dtype=np.complex64)
mean_angle_list = np.zeros((num_samples, num_clusters))

# normalization_vector = np.ones(Lp)/np.sqrt(num_subpaths)
normalization_vector = np.ones(Lp)/np.sqrt(Lp)

for i in range(num_samples):
    if i%100 == 0:
        print('Sample %d'%i)
    path_gains = np.sqrt(1 / 2) * (np.random.randn(Lp) + 1j * np.random.randn(Lp))
    taus = np.zeros(Lp)
    normalized_AoAs = np.zeros(Lp)
    normalized_AoDs = np.zeros(Lp)

    for nc in range(num_clusters):
        # truncated laplacian distribution
        mean_AoA = np.random.uniform(0,360)
        mean_AoD = np.random.uniform(0,360)
        
        mean_angle_list[i,nc] = mean_AoD / 180 * np.pi
        
        AoAs = np.random.laplace(loc=mean_AoA, scale=b_angle, size=num_subpaths) 
        AoAs = np.maximum(AoAs, mean_AoA-2*AS)
        AoAs = np.minimum(AoAs, mean_AoA + 2 * AS)
        AoAs = AoAs / 180 * np.pi
        
        AoDs = np.random.laplace(loc=mean_AoD, scale=b_angle, size=num_subpaths) 
        AoDs = np.maximum(AoDs, mean_AoD-2*AS)
        AoDs = np.minimum(AoDs, mean_AoD+2*AS)
        AoDs = AoDs / 180 * np.pi
        
        normalized_AoAs[nc*num_subpaths:(nc+1)*num_subpaths] = np.sin(AoAs) / 2
        normalized_AoDs[nc*num_subpaths:(nc+1)*num_subpaths] = np.sin(AoDs) / 2
        
        mean_tau = np.random.uniform(0, tau_max)
        taus_cluster = np.random.laplace(loc=mean_tau, scale=b_delay, size=num_subpaths)
        taus_cluster = np.maximum(taus_cluster,mean_tau-2*DS)
        taus_cluster = np.minimum(taus_cluster, mean_tau + 2 * DS)
        taus_cluster = np.maximum(taus_cluster, 0)
        taus_cluster = np.maximum(taus_cluster, tau_max)
        taus[nc*num_subpaths:(nc+1)*num_subpaths] = taus_cluster

    for n in range(num_sc):
        fn = fc + eta*(n-(num_sc-1)/2)
        # frequency dependent steering vectors with beam squint
        A_R = np.exp(-2j*np.pi*(fn/fc)*(np.expand_dims(np.arange(num_antenna_ue),axis=-1).dot(np.expand_dims(normalized_AoAs,axis=0))))
        A_T = np.exp(-2j*np.pi*(fn/fc)*(np.expand_dims(normalized_AoDs,axis=-1).dot(np.expand_dims(np.arange(num_antenna_bs),axis=0))))
        scaler_matrix = path_gains*np.exp(-2j*np.pi*fn*taus)*normalization_vector
        H_list[i,n] = (A_R*scaler_matrix).dot(A_T)
    
    # for n in range(num_sc):
    #     fn = fc + eta*(n-(num_sc-1)/2)
    #     # frequency dependent steering vectors with beam squint
    #     h = 0
    #     for l in range(Lp):
    #         a_R = np.exp(-2j*np.pi*(fn/fc)*np.expand_dims(np.arange(num_antenna_ue)*normalized_AoAs[l],axis=-1))
    #         a_T = np.exp(-2j*np.pi*(fn/fc)*np.expand_dims(np.arange(num_antenna_bs)*normalized_AoDs[l],axis=0))
    #         h = h + path_gains[l]*np.exp(-2j*np.pi*fn*taus[l])/np.sqrt(Lp)*(a_R.dot(a_T))                      
    #     H_list[i,n] = h        
        
print(H_list.shape)
print(mean_angle_list.shape)

io.savemat('./data/channel_%dclusters_%dNt_%dNr_%dscs_%dAS.mat'%(num_clusters, num_antenna_bs, num_antenna_ue, num_sc, AS),{'H_list':H_list,'mean_angle_list':mean_angle_list})

from matplotlib import pyplot as plt
plt.plot(np.abs(np.fft.fft(H_list[0,0,0])))
plt.figure()
plt.plot(np.abs(np.fft.fft(H_list[1,0,0])))

