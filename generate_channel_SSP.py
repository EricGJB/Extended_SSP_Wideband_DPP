from scipy import io
import numpy as np

num_samples = 1000
random_seed = 1011
np.random.seed(random_seed)

Nc = 3
Np = 10

# 拉普拉斯分布的方差等于2b^2
AS = 5.0
b_angle = np.sqrt(AS**2/2)
DS = 1*1e-9
b_delay = np.sqrt(DS**2/2)

# system parameters
fc, W, tau_max = 100 * 1e9, 10 * 1e9, 20 * 1e-9
num_antenna_bs, num_antenna_ue = 128, 4

num_sc = 8
eta = W / num_sc
Lp = Nc * Np

H_list = np.zeros((num_samples, num_sc, num_antenna_bs, num_antenna_ue)) + 1j * np.zeros((num_samples, num_sc, num_antenna_bs, num_antenna_ue))

for i in range(num_samples):
    if i%500 == 0:
        print('Sample %d'%i)
    path_gains = np.sqrt(1 / 2) * (np.random.randn(Lp) + 1j * np.random.randn(Lp))
    taus = np.zeros(Lp)
    normalized_AoAs = np.zeros(Lp)
    normalized_AoDs = np.zeros(Lp)

    for nc in range(Nc):
        # truncated laplacian distribution
        mean_AoA = np.random.uniform(0,360)
        mean_AoD = np.random.uniform(0,360)
        AoAs = np.random.laplace(loc=mean_AoA, scale=b_angle, size=Np) / 180 * np.pi
        AoAs = np.maximum(AoAs,mean_AoA-2*AS)
        AoAs = np.minimum(AoAs, mean_AoA + 2 * AS)
        AoDs = np.random.laplace(loc=mean_AoD, scale=b_angle, size=Np) / 180 * np.pi
        AoDs = np.maximum(AoDs,mean_AoD-2*AS)
        AoDs = np.minimum(AoDs, mean_AoD + 2 * AS)
        normalized_AoAs[nc*Np:(nc+1)*Np] = np.sin(AoAs) / 2
        normalized_AoDs[nc*Np:(nc+1)*Np] = np.sin(AoDs) / 2
        mean_tau = np.random.uniform(0, tau_max)
        taus_cluster = np.random.laplace(loc=mean_tau, scale=b_delay, size=Np)
        taus_cluster = np.maximum(taus_cluster,mean_tau-2*DS)
        taus_cluster = np.minimum(taus_cluster, mean_tau + 2 * DS)
        taus_cluster = np.maximum(taus_cluster, 0)
        taus_cluster = np.maximum(taus_cluster, tau_max)
        taus[nc*Np:(nc+1)*Np] = taus_cluster

    for n in range(num_sc):
        fn = fc + n * eta
        # frequency dependent steering vectors with beam squint
        A_R = np.exp(-2j*np.pi*(fn/fc)*(np.expand_dims(np.arange(num_antenna_bs),axis=-1).dot(np.expand_dims(normalized_AoAs,axis=0))))
        A_T = np.exp(-2j*np.pi*(fn/fc)*(np.expand_dims(normalized_AoDs,axis=-1).dot(np.expand_dims(np.arange(num_antenna_ue),axis=0))))
        scaler_matrix = path_gains*np.exp(-2j*np.pi*fn*taus)/np.sqrt(Lp)
        H_list[i,n] = (A_R*scaler_matrix).dot(np.conjugate(A_T))
        
print(H_list.shape)
io.savemat('./data/channel_%dclusters_%dscs_laplace.mat'%(Nc, num_sc),{'H_list':H_list})

from matplotlib import pyplot as plt
plt.plot(np.abs(np.fft.fft(H_list[0,0,:,0])))
plt.figure()
plt.plot(np.abs(np.fft.fft(H_list[1,0,:,0])))
plt.figure()
plt.plot(np.abs(np.fft.fft(H_list[2,0,:,0])))

