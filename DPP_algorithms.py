import numpy as np
np.random.seed(2022)
import time
from scipy import io
from matplotlib import pyplot as plt

from functions import * 

# system parameters
fc, W, num_sc = 100 * 1e9, 10 * 1e9, 128
eta = W / num_sc

num_antenna_bs, num_TTD = 256, 16
num_ps_per_TTD = num_antenna_bs//num_TTD
 
num_antenna_ue = 4
num_stream = num_antenna_ue

num_clusters = 4
num_rf = num_stream #num_clusters

AS = 5

rou_dB = 10 # SNR in dB
rou = 10**(rou_dB/10) # linear SNR

# load testing channel samples
num_samples = 2
dataset = io.loadmat('./data/channel_%dclusters_%dNt_%dNr_%dscs_%dAS.mat'%(num_clusters, num_antenna_bs, num_antenna_ue, num_sc, AS))
H_list_true = dataset['H_list'][:num_samples]

imperfect = False
if imperfect:
    CE_SNR = 10 # dB
    print('Consider Imperfect CSI, %d dB LS CE SNR'%CE_SNR)
    sigma_2_CE = 1/10**(CE_SNR/10)
    noise_list = np.sqrt(sigma_2_CE/2)*(np.random.randn(num_samples, num_sc, num_antenna_ue, num_antenna_bs)+1j*np.random.randn(num_samples, num_sc, num_antenna_ue, num_antenna_bs))
    H_list_est = H_list_true + noise_list
else:
    print('Consider Perfect CSI')
    H_list_est = np.copy(H_list_true)
    
print(H_list_true.shape)  # (num_samples, num_sc,num_antenna_ue, num_antenna_bs)
print(H_list_est.shape)  # (data_num, num_sc, num_antenna_ue, num_antenna_bs) 



#%% Fully digital beamforming, as upper bound, also the input of matrix decomposition based hybrid precoding algorithms
performance_list = np.zeros(num_samples)
FDBs_list = np.zeros((num_samples,num_sc,num_antenna_bs,num_stream))+1j*np.zeros((num_samples,num_sc,num_antenna_bs,num_stream))

# 0 is just svd
# 1 is waterfilling power allocation among data streams within each subcarrier
# 2 is further waterfilling power allocation among subcarriers, which is not considered here 
# water_filling_level = 1

start = time.time()
for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list = []

    for n in range(num_sc):
        # SVD decomposition, rule in numpy: A = U * Sigma * V        
        H_subcarrier_true = H_list_true[i,n]
        H_subcarrier_est = H_list_est[i,n]  
        
        U, Sigma, V = np.linalg.svd(H_subcarrier_est)
            
        fully_digital_beamformer = np.transpose(np.conjugate(V))
        # no need for normalization since already satisfied
        fully_digital_beamformer = fully_digital_beamformer[:,:num_stream]

        # improve performance by further water-filling power allocation within each subcarrier
        # if water_filling_level == 1:
        eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
        eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
        flag = 1
        num_subchannels = num_stream
        while flag:
            water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
            if water_level > np.max(eq_noise):
                flag = 0
            else:
                eq_noise = eq_noise[:-1]
                num_subchannels = num_subchannels - 1
        pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
        # print(pa_vector)
        fully_digital_beamformer = fully_digital_beamformer * np.sqrt(np.expand_dims(pa_vector, axis=0))
        # print(np.linalg.norm(fully_digital_beamformer)**2)

        # compute the rate, without combiner (won't affect the results)
        temp = H_subcarrier_true.dot(fully_digital_beamformer)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        rate_subcarrier_list.append(np.real(rate_subcarrier))

        FDBs_list[i, n] = fully_digital_beamformer

    performance_list[i] = np.mean(rate_subcarrier_list)

end = time.time()
total_time = end - start

print('Mean time of FDB: %.1f ms'%(total_time/num_samples*1000))
print('Performance of the FDB upper bound: %.4f\n'%np.mean(performance_list))

# print(np.sum(pa_vector))


#%% some frequently used variables 
G_angle = 4*num_antenna_bs # or x8, it's a trade-off between performance and complexity 

sin_value_sc0 = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)

A_list_ideal = []
for n in range(num_sc):
    fn = fc + (n-(num_sc-1)/2)*eta
    sin_value_scn = sin_value_sc0*(fn/fc) 
    # sin_value_scn = sin_value_sc0 # frequency-independent measurement matrices
    A_list_ideal.append(dictionary_angle(num_antenna_bs, G_angle, sin_value_scn))
A_list_ideal = np.array(A_list_ideal)

water_filling_svd = True # If False, pseudo inverse to approximate the fully digital precoder


#%% heuristic DPP, based on dll's paper (same performance as the open source code, but easier to understand)
PS_index_list = (2*np.arange(num_TTD)+1)*num_ps_per_TTD//2
factor = PS_index_list/(2*fc)

## use mean angles of clusters (assume known in simulations)
mean_angle_list = dataset['mean_angle_list'][:num_samples]

performance_list_heuristic = np.zeros(num_samples)

start = time.time()
for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    
    Theta = mean_angle_list[i]
    A = np.zeros((num_antenna_bs,num_rf),dtype=np.complex64)
    T = np.zeros((num_TTD,num_rf))
    
    for j in range(num_rf):
        sin_value = -np.sin(Theta[j])
        a = np.exp(-1j * np.pi * sin_value * np.arange(num_antenna_bs))
        t = sin_value*factor
        A[:,j] = a
        T[:,j] = t
        
    A = A * np.kron(np.exp(1j * 2 * np.pi * fc * T), np.ones((num_ps_per_TTD, 1)))
     
    rate_subcarrier_list = []

    for n in range(num_sc):
        fn = fc + (n-(num_sc-1)/2)*eta 
        Tm = np.exp(-1j * 2 * np.pi * fn * T)
        Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))

        H_subcarrier_true = H_list_true[i,n]
        H_subcarrier_est = H_list_est[i,n] 

        if water_filling_svd:
            # SVD of equivalent channel with water-filling
            # eigen decomposition
            Sigma,U = np.linalg.eig(np.transpose(np.conjugate(Am)).dot(Am))
            eigen_values_normalizer = np.sqrt(1/np.real(Sigma))
            normalizer = (U.dot(np.diag(eigen_values_normalizer))).dot(np.transpose(np.conjugate(U)))
        
            H2_temp = H_subcarrier_est.dot(Am)
            H2_temp = H2_temp.dot(normalizer)
            Ubb,Sigma,Vbb = np.linalg.svd(H2_temp)
            V = np.transpose(np.conjugate(Vbb))
            Fbb = V[:,:num_rf]

            eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
            eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
            flag = 1
            num_subchannels = num_stream
            while flag:
                water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
                if water_level > np.max(eq_noise):
                    flag = 0
                else:
                    eq_noise = eq_noise[:-1]
                    num_subchannels = num_subchannels - 1
            pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
            
            Fbb = Fbb * np.sqrt(np.expand_dims(pa_vector, axis=0))
            Fbb = normalizer.dot(Fbb)
            
            HBF = Am.dot(Fbb)

        else:
            # matrix approximation
            HBFs = np.zeros((num_antenna_bs,num_stream),dtype=np.complex64)
            F_BB_n = np.linalg.pinv(Am).dot(FDBs_list[i,n])
            HBF = Am.dot(F_BB_n)
            # normalization
            scaler = 1 / np.linalg.norm(HBF) * np.sqrt(num_stream)
            HBF = HBF * scaler 
       
        # print(np.linalg.norm(HBF)**2)

        # compute the rate, without combiner (won't affect the results)
        temp = H_subcarrier_true.dot(HBF)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        rate_subcarrier_list.append(np.real(rate_subcarrier))

    performance_list_heuristic[i] = np.mean(rate_subcarrier_list)

end = time.time()
total_time = end - start

print('Mean time of heuristic DPP with mean angles: %.1f ms'%(total_time/num_samples*1000))
print('Performance of heuristic DPP with mean angles: %.4f\n'%np.mean(performance_list_heuristic))


## use CS algorithms to determine angles for heuristic DPP, especially in cluster channels
start = time.time()
mean_angle_list = np.zeros((num_samples, num_clusters))
for i in range(num_samples):
    H = H_list_est[i] # (num_sc,num_antenna_ue,num_antenna_bs)
    H = np.transpose(H,(0,2,1))
    max_angle_indexes,responses_first = Extended_SSP_wideband_H(H, A_list_ideal, num_sc, num_antenna_bs, num_antenna_ue, num_clusters)
    # max_angle_indexes,responses_first = Extended_SSP_wideband_H(H, A_list_feasible, num_sc, num_antenna_bs, num_antenna_ue, num_clusters)
    # _,max_angle_indexes = TopK(H, A_list_feasible, num_sc, num_antenna_bs, num_antenna_ue, num_clusters, peak_finding=True)
    mean_angle_list[i] = np.arcsin(sin_value_sc0[np.array(max_angle_indexes)])
    
A_list_heuristic = np.zeros((num_samples,num_antenna_bs, num_rf),dtype=np.complex64)
T_list_heuristic = np.zeros((num_samples,num_TTD, num_rf))

performance_list_heuristic = np.zeros(num_samples)

for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    
    Theta = mean_angle_list[i]
    A = np.zeros((num_antenna_bs,num_rf),dtype=np.complex64)
    T = np.zeros((num_TTD,num_rf))
    
    for j in range(num_rf):
        sin_value = -np.sin(Theta[j])
        a = np.exp(-1j * np.pi * sin_value * np.arange(num_antenna_bs))
        t = sin_value*factor
        A[:,j] = a
        T[:,j] = t
        
    A = A * np.kron(np.exp(1j * 2 * np.pi * fc * T), np.ones((num_ps_per_TTD, 1)))
    
    A_list_heuristic[i] = A
    T_list_heuristic[i] = T
     
    rate_subcarrier_list = []

    for n in range(num_sc):
        fn = fc + (n-(num_sc-1)/2)*eta 
        Tm = np.exp(-1j * 2 * np.pi * fn * T)
        Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))

        H_subcarrier_est = H_list_est[i,n]
        H_subcarrier_true = H_list_true[i,n]

        if water_filling_svd:
            # SVD of equivalent channel with water-filling
            # eigen decomposition
            Sigma,U = np.linalg.eig(np.transpose(np.conjugate(Am)).dot(Am))
            eigen_values_normalizer = np.sqrt(1/np.real(Sigma))
            normalizer = (U.dot(np.diag(eigen_values_normalizer))).dot(np.transpose(np.conjugate(U)))
        
            H2_temp = H_subcarrier_est.dot(Am)
            H2_temp = H2_temp.dot(normalizer)
            Ubb,Sigma,Vbb = np.linalg.svd(H2_temp)
            V = np.transpose(np.conjugate(Vbb))
            Fbb = V[:,:num_rf]

            eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
            eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
            flag = 1
            num_subchannels = num_stream
            while flag:
                water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
                if water_level > np.max(eq_noise):
                    flag = 0
                else:
                    eq_noise = eq_noise[:-1]
                    num_subchannels = num_subchannels - 1
            pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
            
            Fbb = Fbb * np.sqrt(np.expand_dims(pa_vector, axis=0))
            Fbb = normalizer.dot(Fbb)
            
            HBF = Am.dot(Fbb)
            
        else:
            # matrix approximation
            HBFs = np.zeros((num_antenna_bs,num_stream),dtype=np.complex64)
            F_BB_n = np.linalg.pinv(Am).dot(FDBs_list[i,n])
            HBF = Am.dot(F_BB_n)
            # normalization
            scaler = 1 / np.linalg.norm(HBF) * np.sqrt(num_stream)
            HBF = HBF * scaler 
       
        # print(np.linalg.norm(HBF)**2)

        # compute the rate, without combiner (won't affect the results)
        temp = H_subcarrier_true.dot(HBF)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        rate_subcarrier_list.append(np.real(rate_subcarrier))

    performance_list_heuristic[i] = np.mean(rate_subcarrier_list)

end = time.time()
total_time = end - start

print('Mean time of heuristic DPP with CS angles: %.1f ms'%(total_time/num_samples*1000))
print('Performance of heuristic DPP with CS angles: %.4f\n'%np.mean(performance_list_heuristic))



#%% Extended SSP 

##with ideal codebooks
# performance_list_SSP_ideal = []
# total_time = 0
# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     start = time.time()
#     HBFs,F_BBs,_,__ = Extended_SSP_wideband(FDBs, A_list_ideal, num_sc, num_antenna_bs, num_stream, num_rf)
#     end = time.time()
#     total_time = total_time + end - start

#     sum_rate = 0
#     # Performance evaluation
#     for n in range(num_sc):
#         H_subcarrier = H_list[i, n]
#         temp = H_subcarrier.dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         sum_rate = sum_rate + np.real(rate_subcarrier)

#     performance_list_SSP_ideal.append(sum_rate/num_sc)

# print('Mean time of SSP: %.1f ms'%(total_time/num_samples*1000))
# print('Performance of SSP ideal: %.4f\n'%np.mean(performance_list_SSP_ideal))


## with feasible codebooks
F_RF = dictionary_angle(num_antenna_bs, G_angle, sin_value_sc0)

PS_index_list = (2*np.arange(num_TTD)+1)*num_ps_per_TTD//2
factor = PS_index_list/(2*fc)
# limit T's range to [0,t_max]
T = np.expand_dims(sin_value_sc0,axis=-1).dot(np.expand_dims(factor,axis=0)) 
max_delay = (num_antenna_bs - 1) / (2 * fc) 
for i in range(G_angle):
    t = T[i]
    if np.max(t)<0:
        # t = t - np.min(t)
        t = t + max_delay
        T[i] = t
# print(np.max(T))
# print(np.min(T))

A_list_feasible = []
for n in range(num_sc):
    f = (n-(num_sc-1)/2)*eta # ignore the fixed frequency bias fc here to obtain simple solutions
    A_list_feasible.append(np.transpose(F_RF) * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((1,num_ps_per_TTD))))
A_list_feasible = np.transpose(A_list_feasible,(0,2,1))

# modify the A_list_ideal accordingly to have a small mse, but actually it does not really matter
# for n in range(num_sc):
#     fn = (n-(num_sc-1)/2)*eta
#     delta_phase = np.exp(-1j * 2 * np.pi * fn * max_delay)
#     A_list_ideal[n,:,:G_angle//2] = A_list_ideal[n,:,:G_angle//2]*delta_phase
# initial_mse = np.linalg.norm(A_list_feasible-A_list_ideal)**2/np.product(A_list_ideal.shape)

# no need for codebook optimization by AM, the hand-crafted one is good
A_feasible = F_RF
T_feasible = np.transpose(T)    
A_feasible = A_feasible * np.kron(np.exp(1j * 2 * np.pi * fc * T_feasible), np.ones((num_ps_per_TTD, 1)))

max_angle_indexes_list_feasible = np.zeros((num_samples,num_rf))
responses_list = np.zeros((num_samples,G_angle))

performance_list_SSP_feasible = []
mse_list_SSP_feasible = []

A_list_SSP = np.zeros((num_samples,num_antenna_bs, num_rf),dtype=np.complex64)
T_list_SSP = np.zeros((num_samples,num_TTD, num_rf))

# F_BBs_list = np.zeros((num_samples,num_sc,G_angle,num_stream),dtype=np.complex64) 
total_time = 0
for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
    # Optimization
    start = time.time()
    
    # HBFs,max_angle_indexes,responses = Extended_SSP_wideband(FDBs, A_list_ideal, num_sc, num_antenna_bs, num_stream, num_rf)
    HBFs,max_angle_indexes,responses = Extended_SSP_wideband(FDBs, A_list_feasible, num_sc, num_antenna_bs, num_stream, num_rf) 
    end = time.time()
    total_time = total_time + end - start
    # F_BBs_list[i,:,np.array(max_angle_indexes)] = np.transpose(F_BBs,(1,0,2))

    mse_SSP = np.linalg.norm(HBFs-FDBs)**2/np.product(HBFs.shape)
    mse_list_SSP_feasible.append(mse_SSP)

    max_angle_indexes_list_feasible[i] = max_angle_indexes
    responses_list[i] = responses
    
    A_list_SSP[i] = A_feasible[:,max_angle_indexes]
    T_list_SSP[i] = T_feasible[:,max_angle_indexes]

    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier_est = H_list_est[i, n]
        H_subcarrier_true = H_list_true[i,n]
        
        if water_filling_svd:
            # SVD of equivalent channel with water-filling
            # eigen decomposition
            Am = np.transpose(A_list_feasible[n,:,max_angle_indexes])
            Sigma,U = np.linalg.eig(np.transpose(np.conjugate(Am)).dot(Am))
            eigen_values_normalizer = np.sqrt(1/np.real(Sigma))
            normalizer = (U.dot(np.diag(eigen_values_normalizer))).dot(np.transpose(np.conjugate(U)))
        
            H2_temp = H_subcarrier_est.dot(Am)
            H2_temp = H2_temp.dot(normalizer)
            Ubb,Sigma,Vbb = np.linalg.svd(H2_temp)
            V = np.transpose(np.conjugate(Vbb))
            Fbb = V[:,:num_rf]

            eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
            eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
            flag = 1
            num_subchannels = num_stream
            while flag:
                water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
                if water_level > np.max(eq_noise):
                    flag = 0
                else:
                    eq_noise = eq_noise[:-1]
                    num_subchannels = num_subchannels - 1
            pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
            
            Fbb = Fbb * np.sqrt(np.expand_dims(pa_vector, axis=0))
            Fbb = normalizer.dot(Fbb)
            
            HBF = Am.dot(Fbb)
            
        else:
            HBF = HBFs[n]
            
        temp = H_subcarrier_true.dot(HBF)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_SSP_feasible.append(sum_rate/num_sc)


print('Mean time of E-SSP: %.1f ms'%(total_time/num_samples*1000))
# print('Number of different delays in the feasible delay codebook: %d'%len(np.unique(T)))  
# print('MSE of measurement matrix approximation with feasible codebooks: %.4f'%initial_mse)    
# print('FDB approximation mse of SSP feasible: %.5f'%np.mean(mse_list_SSP_feasible))
print('Performance of E-SSP: %.4f\n'%np.mean(performance_list_SSP_feasible))

# plt.figure()
# plt.plot(np.reshape(np.transpose(T),-1))

# plt.figure()
# plt.plot(responses)
# print(max_angle_indexes)


#%% Low-complexity versions

# top K selection 

# peak_finding = False

# max_angle_indexes_list_predicted = np.zeros((num_samples,num_rf))

# performance_list_SSP_low_complexity = []
# total_time = 0
# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     start = time.time()
#     HBFs,max_angle_indexes = TopK(FDBs, A_list_feasible, num_sc, num_antenna_bs, num_stream, num_rf, peak_finding)
#     max_angle_indexes_list_predicted[i] = max_angle_indexes
    
#     end = time.time()
#     total_time = total_time + end - start

#     sum_rate = 0
#     # Performance evaluation
#     for n in range(num_sc):
#         H_subcarrier = H_list_true[i, n]
#         temp = H_subcarrier.dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         sum_rate = sum_rate + np.real(rate_subcarrier)

#     performance_list_SSP_low_complexity.append(sum_rate/num_sc)
    
# print('Performance of SSP topK: %.4f\n'%np.mean(performance_list_SSP_low_complexity))


# # top K selection (among peaks only) 

# peak_finding = True
# max_angle_indexes_list_peak = np.zeros((num_samples,num_rf))

# performance_list_SSP_peak_finder = []
# total_time = 0
# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     start = time.time()
#     HBFs,max_angle_indexes = TopK(FDBs, A_list_feasible, num_sc, num_antenna_bs, num_stream, num_rf, peak_finding)
#     max_angle_indexes_list_peak[i] = max_angle_indexes
    
#     end = time.time()
#     total_time = total_time + end - start
    
#     sum_rate = 0
#     # Performance evaluation
#     for n in range(num_sc):
#         H_subcarrier = H_list_true[i, n]
#         temp = H_subcarrier.dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         sum_rate = sum_rate + np.real(rate_subcarrier)

#     performance_list_SSP_peak_finder.append(sum_rate/num_sc)

# error = 0
# for i in range(num_samples):
#     for j in range(num_rf):
#         if max_angle_indexes_list_peak[i,j] not in max_angle_indexes_list_feasible[i]:
#             error = error + 1
# accuracy = 1-error/num_samples/num_rf

# print('Support accuracy of topK among peaks: %.3f'%accuracy)
# print('Mean time of SSP topK among peaks: %.1f ms'%(total_time/num_samples*1000))
# print('Performance of SSP topK among peaks: %.4f\n'%np.mean(performance_list_SSP_peak_finder))


## top K selection (among peaks only) + hierarchical codebooks
# max_angle_indexes_list_peak_v2 = np.zeros((num_samples,num_rf))

# performance_list_SSP_peak_finder_v2 = []
# total_time = 0

## if AS == 0:
##     down_sample_factor = 4
## else:
##     down_sample_factor = 2
# down_sample_factor = 2
# expand_factor = 2

# A_list_SSP_low_complexity = np.zeros((num_samples,num_antenna_bs, num_rf),dtype=np.complex64)
# T_list_SSP_low_complexity = np.zeros((num_samples,num_TTD, num_rf))

# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     start = time.time()
    
#     # stage 1, coarse grids, find the position of peaks   
#     max_angle_indexes = stage_1(FDBs, A_list_feasible[:,:,::down_sample_factor], num_sc, num_antenna_bs, num_stream, num_rf)
#     max_angle_indexes_expanded = expand_indexes(max_angle_indexes,down_sample_factor,expand_factor,G_angle)
#     HBFs,max_angle_indexes = Hierarchical(FDBs, A_list_feasible[:,:,max_angle_indexes_expanded], A_list_feasible, num_sc, num_antenna_bs, num_stream, num_rf,max_angle_indexes_expanded)
#     max_angle_indexes_list_peak_v2[i] = max_angle_indexes
    
#     end = time.time()
#     total_time = total_time + end - start
#     # print(HBFs)
    
#     A_list_SSP_low_complexity[i] = A_feasible[:,max_angle_indexes]
#     T_list_SSP_low_complexity[i] = T_feasible[:,max_angle_indexes]
        
#     sum_rate = 0
#     # Performance evaluation
#     for n in range(num_sc):
#         H_subcarrier = H_list_true[i, n]
#         temp = H_subcarrier.dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         sum_rate = sum_rate + np.real(rate_subcarrier)

#     performance_list_SSP_peak_finder_v2.append(sum_rate/num_sc)

# error = 0
# for i in range(num_samples):
#     for j in range(num_rf):
#         if max_angle_indexes_list_peak_v2[i,j] not in max_angle_indexes_list_feasible[i]:
#             error = error + 1
# accuracy = 1-error/num_samples/num_rf

# print('Support accuracy of topK among peaks hierarchical: %.3f'%accuracy)
# print('Mean time of SSP topK among peaks hierarchical: %.1f ms'%(total_time/num_samples*1000))
# print('Performance of SSP topK among peaks hierarchical: %.4f\n'%np.mean(performance_list_SSP_peak_finder_v2))



#%% Low-complexity technique 3: partial scs exploitation 
num_sc_used = 8 # or even smaller 
resolution = num_sc//num_sc_used

# if resolution>1:

max_angle_indexes_list_peak_v2_partial_sc = np.zeros((num_samples,num_rf))

performance_list_SSP_peak_finder_v2_partial_sc = []
total_time = 0

down_sample_factor = 2 #4
expand_factor = 2 #1

A_list_SSP_partial_sc = np.zeros((num_samples,num_antenna_bs, num_rf),dtype=np.complex64)
T_list_SSP_partial_sc = np.zeros((num_samples,num_TTD, num_rf))

for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
    # Optimization
    start = time.time()
    
    # stage 1, coarse grids, find the position of peaks   
    max_angle_indexes = stage_1(FDBs[::resolution], A_list_feasible[::resolution,:,::down_sample_factor], num_sc_used, num_antenna_bs, num_stream, num_rf)
    max_angle_indexes_expanded = expand_indexes(max_angle_indexes,down_sample_factor,expand_factor,G_angle)
    max_angle_indexes = stage_2(FDBs[::resolution], A_list_feasible[::resolution,:,max_angle_indexes_expanded], num_sc_used, num_antenna_bs, num_stream, num_rf,max_angle_indexes_expanded,G_angle)
    max_angle_indexes_list_peak_v2_partial_sc[i] = max_angle_indexes
            
    HBFs = stage_3(FDBs,A_list_feasible,max_angle_indexes,num_sc,num_antenna_bs,num_stream)
    
    end = time.time()
    total_time = total_time + end - start
    # print(HBFs)
    
    A_list_SSP_partial_sc[i] = A_feasible[:,max_angle_indexes]
    T_list_SSP_partial_sc[i] = T_feasible[:,max_angle_indexes]

    
    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier_est = H_list_est[i, n]
        H_subcarrier_true = H_list_true[i,n]
        
        if water_filling_svd:
            # SVD of equivalent channel with water-filling
            # eigen decomposition
            Am = np.transpose(A_list_feasible[n,:,max_angle_indexes])
            Sigma,U = np.linalg.eig(np.transpose(np.conjugate(Am)).dot(Am))
            eigen_values_normalizer = np.sqrt(1/np.real(Sigma))
            normalizer = (U.dot(np.diag(eigen_values_normalizer))).dot(np.transpose(np.conjugate(U)))
        
            H2_temp = H_subcarrier_est.dot(Am)
            H2_temp = H2_temp.dot(normalizer)
            Ubb,Sigma,Vbb = np.linalg.svd(H2_temp)
            V = np.transpose(np.conjugate(Vbb))
            Fbb = V[:,:num_rf]

            eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
            eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
            flag = 1
            num_subchannels = num_stream
            while flag:
                water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
                if water_level > np.max(eq_noise):
                    flag = 0
                else:
                    eq_noise = eq_noise[:-1]
                    num_subchannels = num_subchannels - 1
            pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
            
            Fbb = Fbb * np.sqrt(np.expand_dims(pa_vector, axis=0))
            Fbb = normalizer.dot(Fbb)
            
            HBF = Am.dot(Fbb)
            
        else:
            HBF = HBFs[n]
            
        temp = H_subcarrier_true.dot(HBF)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)    

    performance_list_SSP_peak_finder_v2_partial_sc.append(sum_rate/num_sc)
    
print('Mean time of LCE-SSP: %.1f ms'%(total_time/num_samples*1000))
print('Performance of LCE-SSP: %.4f\n'%np.mean(performance_list_SSP_peak_finder_v2_partial_sc))

# print(max_angle_indexes)



#%% alternative optimization-based algorithm, based on Linglong dai's paper
# TTD's phases could be proportional to the RF or baseband subcarriers' frequencies
max_delay = (num_antenna_bs - 1) / (2 * fc) 
# max_delay = num_antenna_bs/fc
# max_delay = 1/eta/2 # 整数倍的情况下, 可以保证FFT based计算结果和非FFD based一模一样
grids = 256
num_max_iter = 20

FFT_based = True
partial_fft = True

if FFT_based:
    assert max_delay <= 1/eta
    grids = int(np.ceil(grids*(1/eta/max_delay)))

## performance of AM with different initialization methods
epsilon = 1e-6

# AM with random initialization
early_stop = True

performance_list_dll = []
total_time = 0
mse_with_iter = 0
total_iters = 0
object_lists = []
HBFs_lists = []
for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list_dll = np.zeros(num_sc)

    random_A = np.random.uniform(-np.pi, np.pi, num_antenna_bs * num_rf)
    random_A = np.exp(1j * random_A)
    random_A = np.reshape(random_A, (num_antenna_bs, num_rf))
    
    random_T = np.random.uniform(0, 1, num_TTD * num_rf) * max_delay
    random_T = np.reshape(random_T, (num_TTD, num_rf))
    
    A_list, T_list, HBFs_list, object_list_dll, time_elapse, iters = AM(early_stop, epsilon, random_A, random_T, FDBs_list[i], fc, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
    
    object_lists.append(object_list_dll)
    HBFs_lists.append(HBFs_list)
    
    total_time = total_time + time_elapse
    if early_stop == False:
        mse_with_iter = mse_with_iter + object_list_dll
    total_iters = total_iters + iters

    HBFs = HBFs_list[-1]
    A = A_list[-1]
    T = T_list[-1]

    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier_est = H_list_est[i, n]
        H_subcarrier_true = H_list_true[i,n]
        
        f = fc + (n-(num_sc-1)/2)*eta
        Tm = np.exp(-1j * 2 * np.pi * f * T)
        Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))
        
        if water_filling_svd:
            # SVD of equivalent channel with water-filling
            # eigen decomposition
            Sigma,U = np.linalg.eig(np.transpose(np.conjugate(Am)).dot(Am))
            eigen_values_normalizer = np.sqrt(1/np.real(Sigma))
            normalizer = (U.dot(np.diag(eigen_values_normalizer))).dot(np.transpose(np.conjugate(U)))
        
            H2_temp = H_subcarrier_est.dot(Am)
            H2_temp = H2_temp.dot(normalizer)
            Ubb,Sigma,Vbb = np.linalg.svd(H2_temp)
            V = np.transpose(np.conjugate(Vbb))
            Fbb = V[:,:num_rf]

            eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
            eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
            flag = 1
            num_subchannels = num_stream
            while flag:
                water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
                if water_level > np.max(eq_noise):
                    flag = 0
                else:
                    eq_noise = eq_noise[:-1]
                    num_subchannels = num_subchannels - 1
            pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
            
            Fbb = Fbb * np.sqrt(np.expand_dims(pa_vector, axis=0))
            Fbb = normalizer.dot(Fbb)
            
            HBF = Am.dot(Fbb)
            
        else:
            HBF = HBFs[n]
            
        temp = H_subcarrier_true.dot(HBF)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_dll.append(sum_rate/num_sc) 
    
print('Number of FFT points: %d'%grids)
print('Mean time of AM: %.1f ms' % (total_time * 1000 / num_samples))
print('Mean number of iterations of AM, with random initialization: %d'%(total_iters/num_samples))
print('Performance of AM, with random initialization: %.4f\n' %np.mean(performance_list_dll))
if early_stop == False:
    io.savemat('./results/%d_%d_random.mat'%(num_antenna_bs,num_antenna_ue),{'objective':mse_with_iter/num_samples})
    # save object_lists 
    np.save('./results/object_lists_random.npy',object_lists)
    np.save('./results/HBFs_lists_random.npy',HBFs_lists)
    

# AM with heuristic initialization
performance_list_dll = []
total_time = 0
mse_with_iter = 0
total_iters = 0
object_lists = []
HBFs_lists = []
for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list_dll = np.zeros(num_sc)

    A_list, T_list, HBFs_list, object_list_dll, time_elapse, iters = AM(early_stop, epsilon, A_list_heuristic[i], T_list_heuristic[i], FDBs_list[i], fc, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
    
    object_lists.append(object_list_dll)
    HBFs_lists.append(HBFs_list)
    
    total_time = total_time + time_elapse
    if early_stop == False:
        mse_with_iter = mse_with_iter + object_list_dll
    total_iters = total_iters + iters

    HBFs = HBFs_list[-1]
    A = A_list[-1]
    T = T_list[-1]

    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier_est = H_list_est[i, n]
        H_subcarrier_true = H_list_true[i,n]
        
        f = fc + (n-(num_sc-1)/2)*eta
        Tm = np.exp(-1j * 2 * np.pi * f * T)
        Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))
        
        if water_filling_svd:
            # SVD of equivalent channel with water-filling
            # eigen decomposition
            Sigma,U = np.linalg.eig(np.transpose(np.conjugate(Am)).dot(Am))
            eigen_values_normalizer = np.sqrt(1/np.real(Sigma))
            normalizer = (U.dot(np.diag(eigen_values_normalizer))).dot(np.transpose(np.conjugate(U)))
        
            H2_temp = H_subcarrier_est.dot(Am)
            H2_temp = H2_temp.dot(normalizer)
            Ubb,Sigma,Vbb = np.linalg.svd(H2_temp)
            V = np.transpose(np.conjugate(Vbb))
            Fbb = V[:,:num_rf]

            eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
            eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
            flag = 1
            num_subchannels = num_stream
            while flag:
                water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
                if water_level > np.max(eq_noise):
                    flag = 0
                else:
                    eq_noise = eq_noise[:-1]
                    num_subchannels = num_subchannels - 1
            pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
            
            Fbb = Fbb * np.sqrt(np.expand_dims(pa_vector, axis=0))
            Fbb = normalizer.dot(Fbb)
            
            HBF = Am.dot(Fbb)
            
        else:
            HBF = HBFs[n]
            
        temp = H_subcarrier_true.dot(HBF)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_dll.append(sum_rate/num_sc) 
    
print('Mean number of iterations of AM, initialized by heuristic DPP with CS angles: %d'%(total_iters/num_samples))
print('Performance of AM, initialized by heuristic DPP: %.4f\n' %np.mean(performance_list_dll))
if early_stop == False:
    io.savemat('./results/%d_%d_heuristic.mat'%(num_antenna_bs,num_antenna_ue),{'objective':mse_with_iter/num_samples})
    # save object_lists 
    np.save('./results/object_lists_heuristic.npy',object_lists)
    np.save('./results/HBFs_lists_heuristic.npy',HBFs_lists)



# AM with E-SSP initialization
performance_list_dll = []
total_time = 0
mse_with_iter = 0
total_iters = 0
object_lists = []
HBFs_lists = []
for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list_dll = np.zeros(num_sc)

    A_list, T_list, HBFs_list, object_list_dll, time_elapse, iters = AM(early_stop, epsilon, A_list_SSP[i], T_list_SSP[i], FDBs_list[i], fc, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
    
    object_lists.append(object_list_dll)
    HBFs_lists.append(HBFs_list)
    
    total_time = total_time + time_elapse
    if early_stop == False:
        mse_with_iter = mse_with_iter + object_list_dll
    total_iters = total_iters + iters

    HBFs = HBFs_list[-1]
    A = A_list[-1]
    T = T_list[-1]

    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier_est = H_list_est[i, n]
        H_subcarrier_true = H_list_true[i,n]
        
        f = fc + (n-(num_sc-1)/2)*eta
        Tm = np.exp(-1j * 2 * np.pi * f * T)
        Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))
        
        if water_filling_svd:
            # SVD of equivalent channel with water-filling
            # eigen decomposition
            Sigma,U = np.linalg.eig(np.transpose(np.conjugate(Am)).dot(Am))
            eigen_values_normalizer = np.sqrt(1/np.real(Sigma))
            normalizer = (U.dot(np.diag(eigen_values_normalizer))).dot(np.transpose(np.conjugate(U)))
        
            H2_temp = H_subcarrier_est.dot(Am)
            H2_temp = H2_temp.dot(normalizer)
            Ubb,Sigma,Vbb = np.linalg.svd(H2_temp)
            V = np.transpose(np.conjugate(Vbb))
            Fbb = V[:,:num_rf]

            eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
            eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
            flag = 1
            num_subchannels = num_stream
            while flag:
                water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
                if water_level > np.max(eq_noise):
                    flag = 0
                else:
                    eq_noise = eq_noise[:-1]
                    num_subchannels = num_subchannels - 1
            pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
            
            Fbb = Fbb * np.sqrt(np.expand_dims(pa_vector, axis=0))
            Fbb = normalizer.dot(Fbb)
            
            HBF = Am.dot(Fbb)
            
        else:
            HBF = HBFs[n]
            
        temp = H_subcarrier_true.dot(HBF)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_dll.append(sum_rate/num_sc) 

# print('FDB approximation mse of AM initialized by low-complexity extended SSP: %.5f'%(mse_with_iter[-1]/num_samples))    
print('Mean number of iterations of AM, initialized by E-SSP: %d'%(total_iters/num_samples))
print('Performance of AM, initialized by E-SSP: %.4f\n' %np.mean(performance_list_dll))
if early_stop == False:
    io.savemat('./results/%d_%d_lce.mat'%(num_antenna_bs,num_antenna_ue),{'objective':mse_with_iter/num_samples})
    # save object_lists 
    np.save('./results/object_lists_lce_ssp.npy',object_lists)
    np.save('./results/HBFs_lists_lce_ssp.npy',HBFs_lists)
    
    

# AM with LCE-SSP initialization
performance_list_dll = []
total_time = 0
mse_with_iter = 0
total_iters = 0
object_lists = []
HBFs_lists = []
for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list_dll = np.zeros(num_sc)

    A_list, T_list, HBFs_list, object_list_dll, time_elapse, iters = AM(early_stop, epsilon, A_list_SSP_partial_sc[i], T_list_SSP_partial_sc[i], FDBs_list[i], fc, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
    
    object_lists.append(object_list_dll)
    HBFs_lists.append(HBFs_list)
    
    total_time = total_time + time_elapse
    if early_stop == False:
        mse_with_iter = mse_with_iter + object_list_dll
    total_iters = total_iters + iters

    HBFs = HBFs_list[-1]
    A = A_list[-1]
    T = T_list[-1]

    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier_est = H_list_est[i, n]
        H_subcarrier_true = H_list_true[i,n]
        
        f = fc + (n-(num_sc-1)/2)*eta
        Tm = np.exp(-1j * 2 * np.pi * f * T)
        Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))
        
        if water_filling_svd:
            # SVD of equivalent channel with water-filling
            # eigen decomposition
            Sigma,U = np.linalg.eig(np.transpose(np.conjugate(Am)).dot(Am))
            eigen_values_normalizer = np.sqrt(1/np.real(Sigma))
            normalizer = (U.dot(np.diag(eigen_values_normalizer))).dot(np.transpose(np.conjugate(U)))
        
            H2_temp = H_subcarrier_est.dot(Am)
            H2_temp = H2_temp.dot(normalizer)
            Ubb,Sigma,Vbb = np.linalg.svd(H2_temp)
            V = np.transpose(np.conjugate(Vbb))
            Fbb = V[:,:num_rf]

            eq_noise_0 = num_stream / (rou * Sigma[:num_stream] ** 2)
            eq_noise = num_stream / (rou * Sigma[:num_stream] ** 2)
            flag = 1
            num_subchannels = num_stream
            while flag:
                water_level = (num_stream + np.sum(eq_noise)) / num_subchannels
                if water_level > np.max(eq_noise):
                    flag = 0
                else:
                    eq_noise = eq_noise[:-1]
                    num_subchannels = num_subchannels - 1
            pa_vector = np.maximum(water_level - eq_noise_0, 0)  # (1,num_stream)
            
            Fbb = Fbb * np.sqrt(np.expand_dims(pa_vector, axis=0))
            Fbb = normalizer.dot(Fbb)
            
            HBF = Am.dot(Fbb)
            
        else:
            HBF = HBFs[n]
            
        temp = H_subcarrier_true.dot(HBF)
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_dll.append(sum_rate/num_sc) 

# print('FDB approximation mse of AM initialized by low-complexity extended SSP: %.5f'%(mse_with_iter[-1]/num_samples))    
print('Mean number of iterations of AM, initialized by LCE-SSP: %d'%(total_iters/num_samples))
print('Performance of AM, initialized by LCE-SSP: %.4f\n' %np.mean(performance_list_dll))
if early_stop == False:
    io.savemat('./results/%d_%d_lce.mat'%(num_antenna_bs,num_antenna_ue),{'objective':mse_with_iter/num_samples})
    # save object_lists 
    np.save('./results/object_lists_lce_ssp.npy',object_lists)
    np.save('./results/HBFs_lists_lce_ssp.npy',HBFs_lists)


# AM with multiple random initializations 
# num_inits = 10 # or larger until performance saturates
# early_stop = False
# performance_list_dll_multi_init = np.zeros(num_samples)
# object_list_inits = np.zeros((num_samples,num_inits))
# best_object_list_inits = np.zeros(num_samples)
# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
        
#     best_object_current_init = 1000
    
#     for j in range(num_inits):
        
#         random_A = np.random.uniform(-np.pi, np.pi, num_antenna_bs * num_rf)
#         random_A = np.exp(1j * random_A)
#         random_A = np.reshape(random_A, (num_antenna_bs, num_rf))
        
#         random_T = np.random.uniform(0, 1, num_TTD * num_rf) * max_delay
#         random_T = np.reshape(random_T, (num_TTD, num_rf))
        
#         _, __, HBFs_list_this_init, object_list_dll, ___, ____ = AM(early_stop, epsilon, random_A, random_T, FDBs_list[i], fc, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
#         object_current_init = object_list_dll[-1]
#         object_list_inits[i,j] = object_current_init
#         if object_current_init<best_object_current_init:
#             best_object_current_init = object_current_init
#             HBFs = HBFs_list_this_init[-1]
#     best_object_list_inits[i] = best_object_current_init

#     rate_subcarrier_list_dll = np.zeros(num_sc)
#     for n in range(num_sc):
#         H_subcarrier = H_list[i, n]
#         temp = H_subcarrier.dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         rate_subcarrier_list_dll[n] = np.real(rate_subcarrier)

#     performance_list_dll_multi_init[i] = np.mean(rate_subcarrier_list_dll)

# print('Performance of AM, best among %d random inits without early stop: %.4f\n'%(num_inits,np.mean(performance_list_dll_multi_init)))


#%% AM with partial scs, random initialization  
# num_sc_used = 8
# resolution = num_sc//num_sc_used

# if resolution>1:
    
#     eta_partial = eta*resolution
    
#     max_delay = (num_antenna_bs - 1) / (2 * fc) 
    
#     grids = 256 
    
#     num_max_iter = 20
    
#     FFT_based = True
    
#     if FFT_based:
#         # assert max_delay <= 1/eta_partial
#         grids = int(np.ceil(grids*(1/eta_partial/max_delay)))
#         # assert grids >= num_sc_used
    
#     partial_fft = True
    
#     early_stop = True
    
#     performance_list_dll_partial = np.zeros(num_samples)
    
#     for i in range(num_samples):
#         if i % 50 == 0:
#             print('Testing sample %d' % i)
#         FDBs = FDBs_list[i]
#         rate_subcarrier_list_dll = np.zeros(num_sc)
    
#         # random_A = np.random.uniform(-np.pi, np.pi, num_antenna_bs * num_rf)
#         # random_A = np.exp(1j * random_A)
#         # random_A = np.reshape(random_A, (num_antenna_bs, num_rf))
        
#         # random_T = np.random.uniform(0, 1, num_TTD * num_rf) * max_delay
#         # random_T = np.reshape(random_T, (num_TTD, num_rf))
        
#         # A_list, T_list, HBFs_list_partial, object_list_dll_partial, _, __ = AM(early_stop, epsilon, random_A, random_T, FDBs[::resolution], fc, num_ps_per_TTD, eta_partial, num_sc_used, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
#         A_list, T_list, HBFs_list_partial, object_list_dll_partial, _, __ = AM(early_stop, epsilon, A_list_heuristic[i], T_list_heuristic[i], FDBs[::resolution], fc, num_ps_per_TTD, eta_partial, num_sc_used, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
#         # A_list, T_list, HBFs_list_partial, object_list_dll_partial, _, __ = AM(early_stop, epsilon, A_list_SSP_partial_sc[i], T_list_SSP_partial_sc[i], FDBs[::resolution], fc, num_ps_per_TTD, eta_partial, num_sc_used, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
    
#         A = A_list[-1]
#         T = T_list[-1]
    
#         HBFs = np.zeros((num_sc, num_antenna_bs, num_stream)) + 1j * np.zeros((num_sc, num_antenna_bs, num_stream))
#         for n in range(num_sc):
#             f = fc + (n-(num_sc-1)/2)*eta
#             Tm = np.exp(-1j * 2 * np.pi * f * T)
#             Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))
#             Dm = np.linalg.pinv(Am).dot(FDBs[n])
#             HBF = Am.dot(Dm)
#             # normalization
#             HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
#             HBFs[n] = HBF
            
#         for n in range(num_sc):
#             H_subcarrier = H_list[i, n]
#             temp = H_subcarrier.dot(HBFs[n])
#             temp = temp.dot(np.transpose(np.conjugate(temp)))
#             rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
    
#             rate_subcarrier_list_dll[n] = np.real(rate_subcarrier)
    
#         performance_list_dll_partial[i] = np.mean(rate_subcarrier_list_dll)
    
#     print('Performance of alternative minimization with partial scs: %.4f\n' %np.mean(performance_list_dll_partial))


#%% AM with larger delay and partial scs 
# num_sc_used = 8
# resolution = num_sc//num_sc_used

# eta_partial = eta*resolution

# max_delay = (num_antenna_bs - 1) / (2 * fc) 

# grids = 256 

# num_max_iter = 20

# FFT_based = True

# if FFT_based:
#     assert max_delay <= 1/eta_partial
#     grids = int(np.ceil(grids*(1/eta_partial/max_delay)))
#     assert grids >= num_sc_used

# partial_fft = False

# performance_list_dll_larger_delay_partial = np.zeros(num_samples)

# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i]
#     rate_subcarrier_list_dll = np.zeros(num_sc)

#     A_list, T_list, HBFs_list_partial, object_list_dll_partial, _ = AM(FDBs[::resolution], fc, num_ps_per_TTD, eta_partial, num_sc_used, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)

#     A = A_list[-1]
#     T = T_list[-1]

#     HBFs = np.zeros((num_sc, num_antenna_bs, num_stream)) + 1j * np.zeros((num_sc, num_antenna_bs, num_stream))
#     for n in range(num_sc):
#         f = fc + (n-(num_sc-1)/2)*eta
#         Tm = np.exp(-1j * 2 * np.pi * f * T)
#         Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))
#         Dm = np.linalg.pinv(Am).dot(FDBs[n])
#         HBF = Am.dot(Dm)
#         # normalization
#         HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
#         HBFs[n] = HBF

#     for n in range(num_sc):
#         H_subcarrier = H_list[i, n]
#         temp = H_subcarrier.dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))

#         rate_subcarrier_list_dll[n] = np.real(rate_subcarrier)

#     performance_list_dll_larger_delay_partial[i] = np.mean(rate_subcarrier_list_dll)

# print('Performance of alternative minimization with larger delay range and partial scs: %.4f' %np.mean(performance_list_dll_larger_delay_partial))



#%% Heursitc approach, designed for the path channel model, based on (sorted) path angles, written by dll 
# def GenerateComBeam(num_antenna_bs,num_TTD,beta,alpha):
#     B1 = np.exp(1j*np.pi*np.arange(num_antenna_bs)*alpha)
#     B2 = np.exp(1j*np.pi*np.arange(num_TTD)*beta)
#     num_ps_per_TTD = num_antenna_bs//num_TTD
#     w = np.zeros(num_antenna_bs,dtype=np.complex64)
#     for i in range(num_TTD):
#         w[i*num_ps_per_TTD:(i+1)*num_ps_per_TTD]=B1[i*num_ps_per_TTD:(i+1)*num_ps_per_TTD]*B2[i]
#     return w

# performance_list_heuristic = np.zeros(num_samples)


# # load matlab channel
# # from scipy import io
# # H_list = io.loadmat('H.mat')['H'] # (num_antenna_ue,num_antenna_bs,num_sc)
# # mean_angle_list = io.loadmat('Theta.mat')['Theta']
# # H_list = np.transpose(H_list,(2,0,1))
# # H_list = np.expand_dims(H_list,axis=0)
# # mean_angle_list = np.expand_dims(mean_angle_list,axis=0)


# start = time.time()
# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
    
#     Theta = mean_angle_list[i]
#     Phase = np.zeros((num_rf,num_antenna_bs))
#     Delay = np.zeros(num_rf)
    
#     for j in range(num_rf):
#         Phase[j] = np.arange(num_antenna_bs)*np.pi*np.sin(Theta[j])
#         Delay[j] = num_ps_per_TTD*np.sin(Theta[j])/2
     
#     rate_subcarrier_list = []

#     for n in range(num_sc):
#         fn = fc + (n-(num_sc-1)/2)*eta
#         F2_temp = np.zeros((num_rf,num_antenna_bs),dtype=np.complex64)
#         for j in range(num_rf):
#             beta = 2*(fn/fc-1)*Delay[j]
#             F2_temp[j] = GenerateComBeam(num_antenna_bs,num_TTD,beta,np.sin(Theta[j]))

#         H_subcarrier = H_list[i,n]

#         H2_temp = H_subcarrier.dot(np.transpose(F2_temp))
#         _,__,Vbb = np.linalg.svd(H2_temp)
#         V = np.transpose(np.conjugate(Vbb))
#         Fbb = V[:,:num_rf]
#         F2 = np.transpose(F2_temp).dot(Fbb)
#         HBF = F2/np.linalg.norm(F2)*np.sqrt(num_stream)

#         # print(np.linalg.norm(HBF)**2)

#         # compute the rate, without combiner (won't affect the results)
#         temp = H_subcarrier.dot(HBF)
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         rate_subcarrier_list.append(np.real(rate_subcarrier))

#     performance_list_heuristic[i] = np.mean(rate_subcarrier_list)

# end = time.time()
# total_time = end - start

# print('Mean time of heuristic DPP: %.1f ms'%(total_time/num_samples*1000))
# print('Performance of heuristic DPP: %.4f\n'%np.mean(performance_list_heuristic))



#%% use alternative optimization to obtain better codebooks
# initial_A = F_RF
# initial_T = np.transpose(T)

# def AM_0(initial_A, initial_T, eta, FDBs, num_sc, num_antenna_bs, num_TTD, num_rf, num_max_iter, grids, max_delay, partial_fft):
#     object_list = np.zeros(num_max_iter+1)   
#     HBFs_list = np.zeros((num_max_iter+1,num_sc,num_antenna_bs, num_rf),dtype=np.complex64)
#     A_list = np.zeros((num_max_iter+1,num_antenna_bs, num_rf),dtype=np.complex64)
#     T_list = np.zeros((num_max_iter+1,num_TTD, num_rf),dtype=np.complex64)
    
#     # random initialization
#     # initialize T
#     T = np.random.uniform(0, 1, num_TTD * num_rf) * max_delay
#     T = np.reshape(T, (num_TTD, num_rf))
#     # initialize A
#     A = np.random.uniform(-np.pi, np.pi, num_antenna_bs * num_rf)
#     A = np.exp(1j * A)
#     A = np.reshape(A, (num_antenna_bs, num_rf))
    
#     # initialize with baselines
#     # A = np.copy(initial_A)
#     # T = np.copy(initial_T)
    
#     # initial evaluation
#     HBFs = np.zeros((num_sc, num_antenna_bs, num_rf)) + 1j * np.zeros((num_sc, num_antenna_bs, num_rf))
#     for n in range(num_sc):
#         # notice that, the initial evaluation uses the baseband model if initialized with baselines
#         f = fc + (n-(num_sc-1)/2)*eta
#         HBF = A * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((num_ps_per_TTD, 1)))
#         HBFs[n] = HBF
#     objective = np.linalg.norm(HBFs - FDBs)**2/np.product(FDBs.shape)
#     # print(objective)
    
#     object_list[0] = objective
#     HBFs_list[0] = HBFs
#     A_list[0] = A
#     T_list[0] = T

#     # iterations
#     for i in range(num_max_iter):
#         # print('iteration %d'%i)
#         ######################### update FRF
#         term = 0
#         for n in range(num_sc):
#             f = fc + (n-(num_sc-1)/2)*eta # 理论上加不加fc都行
#             Tm = np.exp(-1j * 2 * np.pi * f * T)
#             # FBB_n = np.eye(num_rf,dtype=np.complex64)
#             term = term + FDBs[n] * np.conjugate(np.kron(Tm, np.ones((num_ps_per_TTD, 1))))
#         A = np.exp(1j * np.angle(term))

#         ######################### update T
#         TAU_list = []
#         for n in range(num_sc):
#             # FBB_n = np.eye(num_rf,dtype=np.complex64)
#             Theta_m = FDBs[n] * np.conjugate(A)
#             TAU = np.zeros((num_TTD, num_rf)) + 1j * np.zeros((num_TTD, num_rf))
#             for l in range(num_TTD):
#                 for k in range(num_rf):
#                     for p in range(num_ps_per_TTD):
#                         TAU[l, k] = TAU[l, k] + Theta_m[l * num_ps_per_TTD + p, k]
#             TAU_list.append(TAU)

#         delay_list = np.arange(grids)/grids/eta
#         stop_index = int(np.ceil(grids*max_delay*eta))
#         # print(stop_index)
#         # print(grids)
#         f_bias = fc - (num_sc-1)/2*eta # the frequency of the first sc
#         factor_list = np.exp(-1j*2*np.pi*f_bias*delay_list)
#         TAU_list = np.array(TAU_list)
#         sequences = np.conjugate(TAU_list)
        
#         for l in range(num_TTD):
#             for k in range(num_rf):
#                 sequence = sequences[:,l,k]
#                 fft_responses = np.fft.fft(sequence,grids)*factor_list
                
#                 if partial_fft:
#                     fft_responses = fft_responses[:stop_index] 
                
#                 T[l,k] = delay_list[np.argmax(np.real(fft_responses))]
        
#         HBFs = np.zeros((num_sc, num_antenna_bs, num_rf),dtype=np.complex64)
#         for n in range(num_sc):
#             f = fc + (n-(num_sc-1)/2)*eta
#             HBF = A * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((num_ps_per_TTD, 1)))
#             HBFs[n] = HBF
#         objective = np.linalg.norm(HBFs - FDBs)**2/np.product(FDBs.shape)
#         # print(objective)
        
#         object_list[i+1] = objective
#         HBFs_list[i+1] = HBFs
#         A_list[i+1] = A
#         T_list[i+1] = T

#     return A_list, T_list, HBFs_list, np.array(object_list)

# max_delay = (num_antenna_bs - 1) / (2 * fc) 
# # max_delay = 1/eta
# grids = 512*4

# num_max_iter = 20

# assert max_delay <= 1/eta
# grids = int(np.ceil(grids*(1/eta/max_delay)))
# assert grids >= num_sc

# partial_fft = True

# codebook_file_name = './data/codebooks_feasible_%dscs_%dgrids.mat'%(num_sc,G_angle)

# A_feasible_iterations, T_feasible_iterations, A_list_feasible_iterations, object_list = AM_0(initial_A, initial_T, eta, A_list_ideal, num_sc, num_antenna_bs, num_TTD, G_angle, num_max_iter, grids, max_delay, partial_fft)
# optimal_iteration_index = np.where((object_list)==np.min(object_list))[0][0]
# # if optimal_iteration_index == 0:
# if np.min(object_list)>=initial_mse:
#     print('No better codebook than the hand-crafted one is found')
#     A_list_feasible = A_list_feasible
#     A_feasible = initial_A
#     T_feasible = initial_T

# else:
#     print('Better codebook is found through optimization')
#     A_list_feasible = A_list_feasible_iterations[optimal_iteration_index]
#     A_feasible = A_feasible_iterations[optimal_iteration_index]
#     T_feasible = T_feasible_iterations[optimal_iteration_index]
#     io.savemat(codebook_file_name,{'A_list_feasible':A_list_feasible,\
#                                             'A_feasible':A_feasible,'T_feasible':T_feasible})

# plt.figure()
# plt.plot(object_list)



#%% ideal freqeuncy-dependent dictionary matrices, but with broadened beams, to deal with the cluster channel model
# A_list_ideal_cluster = np.zeros((num_sc,num_antenna_bs,G_angle),dtype=np.complex64)

# def angle_to_sin(angle_begin,angle_end):
#     angles = np.linspace(angle_begin,angle_end,1000)
#     sin_angles = np.sin(angles)
#     sin_angle_smallest = np.min(sin_angles)
#     sin_angle_largest = np.max(sin_angles)
#     return sin_angle_smallest, sin_angle_largest
    
# for g in range(G_angle):
#     sin_angle_fc = sin_value_sc0[g] 
    
#     for n in range(num_sc):
#         fn = fc + (n-(num_sc-1)/2)*eta
#         sin_angle_n = sin_angle_fc*(fn/fc)
#         if sin_angle_n>1:
#             sin_angle_n = sin_angle_n-2
#         if sin_angle_n<-1:
#             sin_angle_n = sin_angle_n+2
            
#         angle_n = np.arcsin(sin_angle_n) # arcsin(.)取值范围为[-pi/2,pi/2]

#         angle_begin = angle_n-2*AS/180*np.pi
#         angle_end = angle_n+2*AS/180*np.pi
#         sin_angle_smallest, sin_angle_largest = angle_to_sin(angle_begin,angle_end)
#         max_squint = (sin_angle_largest-sin_angle_smallest)/2

#         max_Ms = np.sqrt(num_antenna_bs/max_squint)
#         Ms = np.floor(max_Ms)
#         while (1/Ms)*np.floor(num_antenna_bs/Ms)<max_squint:
#             Ms = Ms-1
#         V = np.floor(num_antenna_bs/Ms)
#         Ms = int(Ms)
#         V = int(V)
#         # if V==1:
#         #     analog_beamformer = np.exp(-1j * np.pi * np.arange(num_antenna_bs) * sin_angle_n)
#         # else:
            
#         normalized_AoA_subarray_list = sin_angle_smallest/2 + (2*(np.arange(V)+1)-1)/(2*Ms) 
#         fai_subarray_list = np.pi/Ms*np.arange(V)
#         analog_beamformer = np.zeros(num_antenna_bs)+1j*np.zeros(num_antenna_bs)
#         for v in range(V):
#             for m in range(Ms):
#                 analog_beamformer[v*Ms+m] = np.exp(-1j*2*np.pi*(v*Ms+m)*normalized_AoA_subarray_list[v])\
#                                                         *np.exp(-1j*fai_subarray_list[v])
#         A_list_ideal_cluster[n,:,g] = analog_beamformer

    # print(Ms)
    # print(V)
    

# performance_list_SSP_ideal_cluster = []
# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     HBFs,_,__ = Extended_SSP_wideband(FDBs, A_list_ideal_cluster, num_sc, num_antenna_bs, num_stream, num_rf)

#     sum_rate = 0
#     # Performance evaluation
#     for n in range(num_sc):
#         H_subcarrier = H_list[i, n]
#         temp = H_subcarrier.dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         sum_rate = sum_rate + np.real(rate_subcarrier)

#     performance_list_SSP_ideal.append(sum_rate/num_sc)

# print('Performance of SSP upper bound with ideal cluster frequency-dependent measurement matrices: %.4f\n'%np.mean(performance_list_SSP_ideal))

# plt.plot(__,label='cluster')
# plt.legend()


# ## observe the broadened beam pattern
# g = 512
# beam_path = np.exp(-1j*np.pi*sin_value_sc0[g]*np.expand_dims(np.arange(num_antenna_bs),axis=0))
# beam_cluster = A_list_ideal_cluster[0:1,:,g]
# channels = np.exp(-1j*np.pi*np.expand_dims(np.arange(num_antenna_bs),axis=-1).dot(np.expand_dims(sin_value_sc0,axis=0)))
# responses_path = np.squeeze(np.abs(beam_path.dot(channels)))
# responses_cluster = np.squeeze(np.abs(beam_cluster.dot(channels)))
# plt.figure()
# plt.plot(responses_path,label='path beam responses')
# plt.plot(responses_cluster,label='cluster beam responses')
# plt.legend()



#%% AM with larger delay 
# partial_fft = False

# performance_list_dll_larger_delay = np.zeros((num_samples, num_max_iter))

# mse_with_iter = 0
# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
#     rate_subcarrier_list_dll = np.zeros((num_max_iter, num_sc))
#     # dll baseline
#     A_list, T_list, HBFs_list, object_list_dll, _ = AM(FDBs_list[i], fc, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
#     # A_list, T_list, HBFs_list, object_list_dll, _ = AM_with_init(A_list_SSP[i], T_list_SSP[i], FDBs_list[i], fc, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
#     mse_with_iter = mse_with_iter + object_list_dll

#     for it in range(num_max_iter):
#         HBFs = HBFs_list[it]
#         for n in range(num_sc):
#             H_subcarrier = H_list[i, n]
#             temp = H_subcarrier.dot(HBFs[n])
#             temp = temp.dot(np.transpose(np.conjugate(temp)))
#             rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))

#             rate_subcarrier_list_dll[it, n] = np.real(rate_subcarrier)

#     performance_list_dll_larger_delay[i] = np.mean(rate_subcarrier_list_dll, axis=-1)

# print('Performance of alternative minimization with larger delay range: %.4f' %np.mean(np.max(performance_list_dll_larger_delay, axis=-1)))

# print('FDB approximation mse of alternative minimization with larger delay range: %.5f\n'%(mse_with_iter[-1]/num_samples))

# plt.figure()
# plt.plot(mse_with_iter/num_samples)
# plt.plot(np.mean(mse_list_SSP_feasible)*np.ones(num_max_iter))
# plt.legend(['AM','Extended SSP'])
# plt.ylabel('FDB approximation mse')



#%% AM with larger delay and multiple initializations 
# partial_fft = False

# num_inits = 10
# performance_list_dll_larger_delay_multi_init = np.zeros(num_samples)
# object_list_inits = np.zeros((num_samples,num_inits))
# best_object_list_inits = np.zeros(num_samples)
# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
        
#     best_object_current_init = 1000
#     for j in range(num_inits):
#         _, __, HBFs_list_this_init, object_list_dll, _ = AM(FDBs_list[i], fc, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft)
#         object_current_init = object_list_dll[-1]
#         object_list_inits[i,j] = object_current_init
#         if object_current_init<best_object_current_init:
#             best_object_current_init = object_current_init
#             HBFs = HBFs_list_this_init[-1]
#         best_object_list_inits[i] = best_object_current_init

#     rate_subcarrier_list_dll = np.zeros(num_sc)
#     for n in range(num_sc):
#         H_subcarrier = H_list[i, n]
#         temp = H_subcarrier.dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         rate_subcarrier_list_dll[n] = np.real(rate_subcarrier)

#     performance_list_dll_larger_delay_multi_init[i] = np.mean(rate_subcarrier_list_dll)

# print('Performance of alternative minimization with larger delay range and multiple inits: %.4f'%np.mean(performance_list_dll_larger_delay_multi_init))
# print('FDB approximation mse of alternative minimization with larger delay range and multiple inits: %.5f\n'%np.mean(best_object_list_inits))
# print(object_list_inits)



#%% impact of measurement matrices
## looks best with G_angle = 512, num_sc=256
# A_common = dictionary_angle(num_antenna_bs, G_angle, sin_value_sc0)
# sample_index = 1 
# FDBs = FDBs_list[sample_index]
# response_list_independent = np.zeros((num_sc,G_angle))
# response_list_dependent = np.zeros((num_sc,G_angle))
# for n in range(num_sc):
#     response_list_dependent[n] = np.linalg.norm(np.matrix(A_list_ideal[n]).H.dot(np.matrix(FDBs[n])),axis=-1)**2
#     response_list_independent[n] = np.linalg.norm(np.matrix(A_common).H.dot(np.matrix(FDBs[n])),axis=-1)**2

# fig = plt.figure()

# ax1 = fig.add_subplot(111)
# ax1.imshow(response_list_independent,cmap='gray_r')
# ax1.set_xlabel('Atom Index')
# ax1.set_ylim([0,num_sc])
# ax1.set_ylabel('Subcarrier Index')

# from textwrap import fill
# label = fill('Average Projection Over All Subcarriers',20)

# average_responses_independent = np.abs(np.mean(response_list_independent,axis=0))
# ax2 = ax1.twinx()
# ax2.plot(average_responses_independent,'b--',label=label)
# ax2.set_ylabel('Average Projection')
# ax2.legend(loc='center')

# fig.savefig('./figures/projections_fid_matrix.eps')


# fig = plt.figure()

# ax1 = fig.add_subplot(111)
# ax1.imshow(response_list_dependent,cmap='gray_r')
# ax1.set_xlabel('Atom Index')
# ax1.set_ylim([0,num_sc])
# ax1.set_ylabel('Subcarrier Index')

# average_responses_independent = np.abs(np.mean(response_list_dependent,axis=0))
# ax2 = ax1.twinx()
# ax2.plot(average_responses_independent,'r--',label=label)
# ax2.set_ylabel('Average Projection')
# ax2.legend(loc='center')

# fig.savefig('./figures/projections_fd_matrices.eps')



#%% plot an example of average projections
# sample_index = 8
# responses = responses_list[sample_index]/num_sc
# max_angle_indexes_feasible = max_angle_indexes_list_feasible[sample_index]
# max_angle_indexes_predicted = max_angle_indexes_list_predicted[sample_index]
# max_angle_indexes_peak = max_angle_indexes_list_peak[sample_index]

# bias = 505

# shifts_feasible = bias*np.ones(G_angle)
# shifts_feasible[max_angle_indexes_feasible.astype(np.int16)] = shifts_feasible[max_angle_indexes_feasible.astype(np.int16)] - bias
# shifts_predicted = bias*np.ones(G_angle)
# shifts_predicted[max_angle_indexes_predicted.astype(np.int16)] = shifts_predicted[max_angle_indexes_predicted.astype(np.int16)] - bias
# shifts_peak = bias*np.ones(G_angle)
# shifts_peak[max_angle_indexes_peak.astype(np.int16)] = shifts_peak[max_angle_indexes_peak.astype(np.int16)] - bias
# responses_feasible = responses + shifts_feasible
# responses_predicted = responses + shifts_predicted
# responses_peak = responses + shifts_peak
# plt.figure()
# plt.plot(np.abs(responses),'k-',label='Initial Projections',linewidth=1)
# plt.plot(np.abs(responses_feasible),'bo',markerfacecolor='white',label='Atoms Iterative')
# plt.plot(np.abs(responses_predicted),'g+',label='Atoms Top-$N_{RF}$')
# plt.plot(np.abs(responses_peak),'rx',label='Atoms Top-$N_{RF}$ from Peaks')
# plt.xlim(0,G_angle)
# plt.ylim(0,bias-100)
# plt.xlabel('Atom Index')
# plt.ylabel('Average Projection')
# plt.legend(loc='upper left',bbox_to_anchor=[0,1],ncol=2,shadow=False,fancybox=False)
# plt.grid(linewidth=0.5)

# plt.savefig('./figures/atom_selection.eps')

# print(max_angle_indexes_peak)
# print(max_angle_indexes_predicted)
# print(max_angle_indexes_feasible)

# print(performance_list_SSP_peak_finder[sample_index])
# print(performance_list_SSP_low_complexity[sample_index])
# print(performance_list_SSP_feasible[sample_index])
# print('\n')



#%% heuristic approach, based on gff's paper
# performance_list_gff = np.zeros(num_samples)

# for i in range(num_samples):
#     if i%200==0:
#         print('Testing sample %d'%i)

#     Theta = mean_angle_list[i]
#     normalized_AoAs = -np.sin(Theta)/2 # due to hermitian

#     matrix_t = np.zeros((num_rf,num_TTD))
#     matrix_Phi = np.zeros((num_rf,num_TTD))
    
#     for l in range(num_rf):
#         normalized_AoA = normalized_AoAs[l]
#         term = normalized_AoA*num_ps_per_TTD/fc
#         for v in range(num_TTD):
#             matrix_t[l,v] = v*term
#             matrix_Phi[l,v] = -v*term*np.pi*W
    
#     F_RF_0 = np.zeros((num_antenna_bs,num_rf))+1j*np.zeros((num_antenna_bs,num_rf))   
#     for l in range(num_rf):
#         normalized_AoA = normalized_AoAs[l]
#         for m in range(num_antenna_bs):
#             F_RF_0[m,l] = np.exp(-1j*(2*np.pi*m*normalized_AoA*(1+W/2/fc)+matrix_Phi[l,int(np.ceil((m+1)/num_ps_per_TTD))-1]))
        
#     # iterate over all subcarriers, get all F_BBs, and compute average rate
#     rate_subcarrier_list = []
#     for n in range(num_sc):
#         f = (n-(num_sc-1)/2)*eta
        
#         H_subcarrier = np.transpose(np.conjugate(H_list[i,n]))
                    
#         Matrix_Fai = np.zeros((num_TTD,num_rf))+1j*np.zeros((num_TTD,num_rf))
#         for v in range(num_TTD):
#             for l in range(num_rf):
#                 Matrix_Fai[v,l]=np.exp(-1j*2*np.pi*f*matrix_t[l,v])
        
#         F_TTD = np.kron(Matrix_Fai, np.ones((num_ps_per_TTD,1)))
            
#         # frequency-dependent F_RF        
#         F_RF = F_RF_0*F_TTD
                    
#         # eigen decomposition
#         Sigma,U = np.linalg.eig(np.transpose(np.conjugate(F_RF)).dot(F_RF))
#         eigen_values_normalizer = np.sqrt(1/np.real(Sigma))
#         # ensure that for each n, np.linalg.norm(F_RF.dot(F_BB))**2 = Ns
#         normalizer = (U.dot(np.diag(eigen_values_normalizer))).dot(np.transpose(np.conjugate(U)))
#         channel_multiplier = F_RF.dot(normalizer)
        
#         H_equivalent_normalized = np.transpose(np.conjugate(H_subcarrier)).dot(channel_multiplier)
                      
#         # SVD decomposition, rule: A = U * Sigma * V
#         U,Sigma,V = np.linalg.svd(H_equivalent_normalized)

#         # obtain F_BB_n, (N_RF, Ns)
#         F_BB = normalizer.dot(np.transpose(np.conjugate(V))[:,:num_stream])
    
#         hybrid_beamformer = F_RF.dot(F_BB)
            
#         # compute the rate
#         channel_with_beamformer = np.transpose(np.conjugate(H_subcarrier)).dot(hybrid_beamformer)
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue)+rou/num_stream*\
#                                   channel_with_beamformer.dot(np.transpose(np.conjugate(channel_with_beamformer)))))
#         rate_subcarrier_list.append(np.real(rate_subcarrier))
    
#     performance_list[i] = np.mean(rate_subcarrier_list)
    
# print('Conventional TTD performance: %.4f\n'%np.mean(performance_list))



#%% directly optimize feasible phase and delay codebooks for cluster channels using AM. the complexity is too high 
# print(FDBs_list.shape)
# print(F_BBs_list.shape)

# def Optimize_Codebooks_AM_with_init(initial_A, initial_T, FDBs_list, F_BBs_list, fc, num_samples, G_angle, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay):
#     # object_list = []
    
#     # must use .copy since T is updated element-wisely 
#     A = np.copy(initial_A)
#     T = np.copy(initial_T) 
    
#     # when the input of initial_A and _T are based on baseband model
#     A = A * np.kron(np.exp(1j * 2 * np.pi * fc * T), np.ones((num_ps_per_TTD, 1))) 
    
#     time_elapse = 0
    
#     pinv_list = np.linalg.pinv(F_BBs_list)
#     norm_list = np.linalg.norm(F_BBs_list,axis=(-1,-2))**2
    
#     for i in range(num_max_iter):
#         print('Iteration %d'%i)
       
#         start = time.time()

#         ######################### update FRF
#         term = 0
#         Theta_list = np.zeros((num_samples,num_sc,num_antenna_bs,G_angle),dtype=np.complex64)
#         for n in range(num_sc):
#             print(n)
#             f = fc + (n-(num_sc-1)/2)*eta
#             Tm = np.exp(-1j * 2 * np.pi * f * T)
#             tmp = np.conjugate(np.kron(Tm, np.ones((num_ps_per_TTD, 1))))
#             for k in range(num_samples):
#                 Theta_k_n = FDBs_list[k,n].dot(pinv_list[k,n])
#                 term = term + norm_list[k,n] * Theta_k_n * tmp
#                 Theta_list[k,n] = Theta_k_n
#         A = np.exp(1j * np.angle(term))

#         ######################### update T            
#         delay_list = np.arange(grids)/grids/eta
#         stop_index = int(np.ceil(grids*max_delay*eta))
#         f_bias = fc - (num_sc-1)/2*eta # frequency of the initial subcarrier
#         factor_list = np.exp(-1j*2*np.pi*f_bias*delay_list)

#         for l in range(num_TTD):
#             print(l)
#             for k in range(G_angle):
                
#                 fft_responses_sum = np.zeros(stop_index)
#                 for kk in range(num_samples):
                    
#                     TAU_list = []
#                     for n in range(num_sc):
#                         Theta_m = Theta_list[kk,n] * np.conjugate(A)
#                         TAU = np.zeros((num_TTD, G_angle)) + 1j * np.zeros((num_TTD, G_angle))
#                         for lll in range(num_TTD):
#                             for kkk in range(G_angle):
#                                 for p in range(num_ps_per_TTD):
#                                     TAU[lll, kkk] = TAU[lll, kkk] + Theta_m[lll * num_ps_per_TTD + p, kkk]
#                         TAU_list.append(TAU)
#                     TAU_list = np.array(TAU_list)
                    
#                     sequences = np.conjugate(TAU_list)*np.expand_dims(np.expand_dims(norm_list[kk],axis=-1),axis=-1)
#                     sequence = sequences[:,l,k]
#                     fft_responses = np.fft.fft(sequence,grids)*factor_list
#                     fft_responses = fft_responses[:stop_index]
#                     fft_responses_sum = fft_responses_sum + fft_responses
                
#                 T[l,k] = delay_list[np.argmax(np.real(fft_responses_sum))]     
            
            
#         end = time.time()
#         time_elapse = time_elapse + end - start

#     return A, T, time_elapse

# initial_A = A_feasible
# initial_T = T_feasible

# num_max_iter = 10
# grids = 256
# grids = int(np.ceil(grids*(1/eta/max_delay)))
# A_optimized, T_optimized, time_elapse = Optimize_Codebooks_AM_with_init(initial_A, initial_T, FDBs_list, F_BBs_list, fc, num_samples, G_angle, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay)



#%% optimize the ideal measurement matrices first, i.e., Ams, using the AM algorithm
# A_list_optimized = np.copy(A_list_ideal)
# num_max_iter = 20

# for it in range(num_max_iter):
#     print('Iteration %d'%it)
    
# ################################ update F_BBs_list
#     F_BBs_list = np.zeros((num_samples,num_sc,G_angle,num_stream),dtype=np.complex64) 
#     object_value = 0
#     for i in range(num_samples):
#         FDBs = FDBs_list[i] 
#         HBFs,F_BBs,max_angle_indexes,responses = Extended_SSP_wideband(FDBs, A_list_optimized, num_sc, num_antenna_bs, num_stream, num_rf)
#         object_value = object_value + np.linalg.norm(HBFs-FDBs_list[i])**2/np.product(HBFs.shape)
#         F_BBs_list[i,:,np.array(max_angle_indexes)] = np.transpose(F_BBs,(1,0,2))
#     print(object_value/num_samples)
    
# ################################ update A_list_optimized
#     pinv_list = np.linalg.pinv(F_BBs_list)
#     norm_list = np.linalg.norm(F_BBs_list,axis=(-1,-2))**2

#     for n in range(num_sc):
#         term = 0
#         for k in range(num_samples):
#             term = term + norm_list[k,n] * FDBs_list[k,n].dot(pinv_list[k,n])
#         A_list_optimized[n] = np.exp(1j*np.angle(term))
    

# ## see the performance of the optimized ideal measurement matrices
# performance_list_SSP_optimized = []
# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     HBFs,F_BBs,max_angle_indexes,responses = Extended_SSP_wideband(FDBs, A_list_optimized, num_sc, num_antenna_bs, num_stream, num_rf)

#     sum_rate = 0
#     # Performance evaluation
#     for n in range(num_sc):
#         H_subcarrier = H_list[i, n]
#         temp = H_subcarrier.dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         sum_rate = sum_rate + np.real(rate_subcarrier)

#     performance_list_SSP_optimized.append(sum_rate/num_sc)

# print('Performance of SSP with optimized ideal matrices for cluster channels: %.4f\n'%np.mean(performance_list_SSP_optimized))


# ## then, approximate the constructed ideal matrices with feasible phase and delay codebooks, still use AM 
# def AM_0(eta, FDBs, num_sc, num_antenna_bs, num_TTD, num_rf, num_max_iter, grids, max_delay, partial_fft):
#     object_list = np.zeros(num_max_iter+1)   
#     HBFs_list = np.zeros((num_max_iter+1,num_sc,num_antenna_bs, num_rf),dtype=np.complex64)
#     A_list = np.zeros((num_max_iter+1,num_antenna_bs, num_rf),dtype=np.complex64)
#     T_list = np.zeros((num_max_iter+1,num_TTD, num_rf),dtype=np.complex64)
    
#     # initialize T
#     T = np.random.uniform(0, 1, num_TTD * num_rf) * max_delay
#     T = np.reshape(T, (num_TTD, num_rf))
#     # initialize A
#     A = np.random.uniform(-np.pi, np.pi, num_antenna_bs * num_rf)
#     A = np.exp(1j * A)
#     A = np.reshape(A, (num_antenna_bs, num_rf))
    
#     # initial evaluation
#     HBFs = np.zeros((num_sc, num_antenna_bs, num_rf)) + 1j * np.zeros((num_sc, num_antenna_bs, num_rf))
#     for n in range(num_sc):
#         # notice that, the initial evaluation uses the baseband model if initialized with baselines
#         f = fc + (n-(num_sc-1)/2)*eta
#         HBF = A * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((num_ps_per_TTD, 1)))
#         HBFs[n] = HBF
#     objective = np.linalg.norm(HBFs - FDBs)**2/np.product(FDBs.shape)
#     # print(objective)
    
#     object_list[0] = objective
#     HBFs_list[0] = HBFs
#     A_list[0] = A
#     T_list[0] = T

#     # iterations
#     for i in range(num_max_iter):
#         # print('iteration %d'%i)
#         ######################### update FRF
#         term = 0
#         for n in range(num_sc):
#             f = fc + (n-(num_sc-1)/2)*eta # 理论上加不加fc都行
#             Tm = np.exp(-1j * 2 * np.pi * f * T)
#             # FBB_n = np.eye(num_rf,dtype=np.complex64)
#             term = term + FDBs[n] * np.conjugate(np.kron(Tm, np.ones((num_ps_per_TTD, 1))))
#         A = np.exp(1j * np.angle(term))

#         ######################### update T
#         TAU_list = []
#         for n in range(num_sc):
#             # FBB_n = np.eye(num_rf,dtype=np.complex64)
#             Theta_m = FDBs[n] * np.conjugate(A)
#             TAU = np.zeros((num_TTD, num_rf)) + 1j * np.zeros((num_TTD, num_rf))
#             for l in range(num_TTD):
#                 for k in range(num_rf):
#                     for p in range(num_ps_per_TTD):
#                         TAU[l, k] = TAU[l, k] + Theta_m[l * num_ps_per_TTD + p, k]
#             TAU_list.append(TAU)

#         delay_list = np.arange(grids)/grids/eta
#         stop_index = int(np.ceil(grids*max_delay*eta))
#         # print(stop_index)
#         # print(grids)
#         f_bias = fc - (num_sc-1)/2*eta # the frequency of the first sc
#         factor_list = np.exp(-1j*2*np.pi*f_bias*delay_list)
#         TAU_list = np.array(TAU_list)
#         sequences = np.conjugate(TAU_list)
        
#         for l in range(num_TTD):
#             for k in range(num_rf):
#                 sequence = sequences[:,l,k]
#                 fft_responses = np.fft.fft(sequence,grids)*factor_list
                
#                 if partial_fft:
#                     fft_responses = fft_responses[:stop_index] 
                
#                 T[l,k] = delay_list[np.argmax(np.real(fft_responses))]
        
#         HBFs = np.zeros((num_sc, num_antenna_bs, num_rf),dtype=np.complex64)
#         for n in range(num_sc):
#             f = fc + (n-(num_sc-1)/2)*eta
#             HBF = A * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((num_ps_per_TTD, 1)))
#             HBFs[n] = HBF
#         objective = np.linalg.norm(HBFs - FDBs)**2/np.product(FDBs.shape)
#         # print(objective)
        
#         object_list[i+1] = objective
#         HBFs_list[i+1] = HBFs
#         A_list[i+1] = A
#         T_list[i+1] = T

#     return A_list, T_list, HBFs_list, np.array(object_list)

# max_delay = (num_antenna_bs - 1) / (2 * fc) 

# grids = 512*4

# num_max_iter = 20

# assert max_delay <= 1/eta
# grids = int(np.ceil(grids*(1/eta/max_delay)))
# assert grids >= num_sc

# partial_fft = True

# A_feasible_iterations, T_feasible_iterations, A_list_feasible_iterations, object_list = AM_0(eta, A_list_optimized, num_sc, num_antenna_bs, num_TTD, G_angle, num_max_iter, grids, max_delay, partial_fft)
# optimal_iteration_index = np.where((object_list)==np.min(object_list))[0][0]

# A_list_feasible_optimized = A_list_feasible_iterations[optimal_iteration_index]

# ## see the performance of the optimized  feasible measurement matrices
# performance_list_SSP_optimized_feasible = []
# for i in range(num_samples):
#     if i % 50 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     HBFs,F_BBs,max_angle_indexes,responses = Extended_SSP_wideband(FDBs, A_list_feasible_optimized, num_sc, num_antenna_bs, num_stream, num_rf)

#     sum_rate = 0
#     # Performance evaluation
#     for n in range(num_sc):
#         H_subcarrier = H_list[i, n]
#         temp = H_subcarrier.dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         sum_rate = sum_rate + np.real(rate_subcarrier)

#     performance_list_SSP_optimized_feasible.append(sum_rate/num_sc)

# print('Performance of SSP with optimized feasible matrices for cluster channels: %.4f\n'%np.mean(performance_list_SSP_optimized_feasible))


