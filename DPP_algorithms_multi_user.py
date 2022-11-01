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
 
num_antenna_ue = 1
num_stream = num_antenna_ue

num_clusters = 4

AS = 5

rou_dB = -10 # SNR in dB
rou = 10**(rou_dB/10) # linear SNR

num_user = 4

num_rf = num_user*num_clusters

# load testing channel samples
num_samples = 200
dataset = io.loadmat('./data/channel_%dclusters_%dNt_%dNr_%dscs_%dAS.mat'%(num_clusters, num_antenna_bs, num_antenna_ue, num_sc, AS))
H_list_true = dataset['H_list'][:num_samples*num_user]

imperfect = False
if imperfect:
    CE_SNR = 10 # dB
    print('Consider Imperfect CSI, %d dB LS CE SNR'%CE_SNR)
    sigma_2_CE = 1/10**(CE_SNR/10)
    noise_list = np.sqrt(sigma_2_CE/2)*(np.random.randn(num_samples*num_user, num_sc, num_antenna_ue, num_antenna_bs)+1j*np.random.randn(num_samples*num_user, num_sc, num_antenna_ue, num_antenna_bs))
    H_list_est = H_list_true + noise_list
else:
    print('Consider Perfect CSI')
    H_list_est = np.copy(H_list_true)
    
H_list_true = np.reshape(H_list_true,(num_samples, num_user, num_sc, num_antenna_ue, num_antenna_bs))
H_list_true = np.transpose(H_list_true,(0,2,3,4,1))
H_list_est = np.reshape(H_list_est,(num_samples, num_user, num_sc, num_antenna_ue, num_antenna_bs))
H_list_est = np.transpose(H_list_est,(0,2,3,4,1))

print(H_list_true.shape)  # (num_samples, num_sc, num_antenna_ue, num_antenna_bs, num_user)
print(H_list_est.shape) # (num_samples, num_sc, num_antenna_ue, num_antenna_bs, num_user) 



#%% upper bound
# num_iter = 50

p = num_stream*num_user

sigma_2 = 1/rou

performance_list_ZF = np.zeros(num_samples)
# performance_list_WMMSE = np.zeros(num_samples)
FDBs_list = np.zeros((num_samples,num_sc,num_antenna_bs,num_stream*num_user))+1j*np.zeros((num_samples,num_sc,num_antenna_bs,num_stream*num_user))

start = time.time()
for i in range(num_samples):
    if i % 50 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list_ZF = []
    # rate_subcarrier_list_WMMSE = []
    
    for n in range(num_sc):
        H_subcarrier_true = H_list_true[i,n]
        H_subcarrier_est = H_list_est[i,n] 

        # ZF
        V = ZF(H_subcarrier_est,num_antenna_bs,num_antenna_ue,num_stream,num_user,p)
        sum_rate = 0
        for k in range(num_user):
            H_k = H_subcarrier_true[:, :, k]  # NrxNt
            V_k = V[:, :, k]  # Ntx1
            signal_k = H_k.dot(V_k)
            signal_k_energy = signal_k.dot(np.transpose(np.conjugate(signal_k)))
            interference_k_energy = 0
            for j in range(num_user):
                if j != k:
                    V_j = V[:, :, j]
                    interference_j = H_k.dot(V_j)
                    interference_k_energy = interference_k_energy + interference_j.dot(np.transpose(np.conjugate(interference_j)))
            SINR_k = signal_k_energy.dot(np.linalg.inv(interference_k_energy + sigma_2 * np.eye(num_antenna_ue)))
            rate_k = np.log2(np.linalg.det(np.eye(num_antenna_ue) + SINR_k))
            sum_rate = sum_rate + rate_k  
        
        rate_subcarrier_list_ZF.append(np.real(sum_rate))
        
        # # WMMSE, initialized by ZF
        # for l in range(num_iter):
        #     U = update_U(H_subcarrier_est,V,num_user,num_antenna_ue,sigma_2,p) # (num_antenna_ue,num_stream,num_user)
        #     W = update_W(H_subcarrier_est,U,V,num_user,num_stream) # (num_stream,num_stream,num_user)
        #     V = update_V(H_subcarrier_est,U,W,num_antenna_bs,num_user,sigma_2,p) # (num_antenna_bs,num_stream,num_user)
                
        # sum_rate = 0
        # for k in range(num_user):
        #     H_k = H_subcarrier_true[:, :, k]  # NrxNt
        #     V_k = V[:, :, k]  # Ntx1
        #     signal_k = H_k.dot(V_k)
        #     signal_k_energy = signal_k.dot(np.transpose(np.conjugate(signal_k)))
        #     interference_k_energy = 0
        #     for j in range(num_user):
        #         if j != k:
        #             V_j = V[:, :, j]
        #             interference_j = H_k.dot(V_j)
        #             interference_k_energy = interference_k_energy + interference_j.dot(np.transpose(np.conjugate(interference_j)))
        #     SINR_k = signal_k_energy.dot(np.linalg.inv(interference_k_energy + sigma_2 * np.eye(num_antenna_ue)))
        #     rate_k = np.log2(np.linalg.det(np.eye(num_antenna_ue) + SINR_k))
        #     sum_rate = sum_rate + rate_k    
            
        # rate_subcarrier_list_WMMSE.append(np.real(sum_rate))
        
        FDBs_list[i, n] = np.reshape(V,(num_antenna_bs,num_stream*num_user))  
        
    performance_list_ZF[i] = np.mean(rate_subcarrier_list_ZF)
    # performance_list_WMMSE[i] = np.mean(rate_subcarrier_list_WMMSE)

end = time.time()
total_time = end - start

# print('Mean time of WMMSE: %.1f ms'%(total_time/num_samples*1000))
print('Performance of ZF: %.4f\n'%np.mean(performance_list_ZF))
# print('Performance of the WMMSE upper bound: %.4f\n'%np.mean(performance_list_WMMSE))

# the number of total streams
num_stream = num_user*num_stream


#%% Extended SSP 
G_angle = 4*num_antenna_bs # or x8, it's a trade-off between performance and complexity 

sin_value_sc0 = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)

A_list_ideal = []
for n in range(num_sc):
    fn = fc + (n-(num_sc-1)/2)*eta
    sin_value_scn = sin_value_sc0*(fn/fc) 
    # sin_value_scn = sin_value_sc0 # frequency-independent measurement matrices
    A_list_ideal.append(dictionary_angle(num_antenna_bs, G_angle, sin_value_scn))
A_list_ideal = np.array(A_list_ideal)


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
    for n in range(num_sc):
        H_subcarrier_true = H_list_true[i,n]
        V = np.expand_dims(HBFs[n],axis=1)
        for k in range(num_user):
            H_k = H_subcarrier_true[:, :, k]  # NrxNt
            V_k = V[:, :, k]  # Ntx1
            signal_k = H_k.dot(V_k)
            signal_k_energy = signal_k.dot(np.transpose(np.conjugate(signal_k)))
            interference_k_energy = 0
            for j in range(num_user):
                if j != k:
                    V_j = V[:, :, j]
                    interference_j = H_k.dot(V_j)
                    interference_k_energy = interference_k_energy + interference_j.dot(np.transpose(np.conjugate(interference_j)))
            SINR_k = signal_k_energy.dot(np.linalg.inv(interference_k_energy + sigma_2 * np.eye(num_antenna_ue)))
            rate_k = np.log2(np.linalg.det(np.eye(num_antenna_ue) + SINR_k))
            sum_rate = sum_rate + rate_k  

    performance_list_SSP_feasible.append(np.real(sum_rate)/num_sc)

print('Mean time of E-SSP: %.1f ms'%(total_time/num_samples*1000))
print('Performance of E-SSP: %.4f\n'%np.mean(performance_list_SSP_feasible))



#%% Low-complexity technique 3: partial scs exploitation 
num_sc_used = 8 # or even smaller 
resolution = num_sc//num_sc_used

if resolution>1:

    max_angle_indexes_list_peak_v2_partial_sc = np.zeros((num_samples,num_rf))
    
    performance_list_SSP_peak_finder_v2_partial_sc = []
    total_time = 0
    
    down_sample_factor = 2
    expand_factor = 2
    
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
            
            sum_rate = 0
            for n in range(num_sc):
                H_subcarrier_true = H_list_true[i,n]
                V = np.expand_dims(HBFs[n],axis=1)
                for k in range(num_user):
                    H_k = H_subcarrier_true[:, :, k]  # NrxNt
                    V_k = V[:, :, k]  # Ntx1
                    signal_k = H_k.dot(V_k)
                    signal_k_energy = signal_k.dot(np.transpose(np.conjugate(signal_k)))
                    interference_k_energy = 0
                    for j in range(num_user):
                        if j != k:
                            V_j = V[:, :, j]
                            interference_j = H_k.dot(V_j)
                            interference_k_energy = interference_k_energy + interference_j.dot(np.transpose(np.conjugate(interference_j)))
                    SINR_k = signal_k_energy.dot(np.linalg.inv(interference_k_energy + sigma_2 * np.eye(num_antenna_ue)))
                    rate_k = np.log2(np.linalg.det(np.eye(num_antenna_ue) + SINR_k))
                    sum_rate = sum_rate + rate_k  
                    
        performance_list_SSP_peak_finder_v2_partial_sc.append(np.real(sum_rate)/num_sc)
        
    print('Mean time of LCE-SSP: %.1f ms'%(total_time/num_samples*1000))
    print('Performance of LCE-SSP: %.4f\n'%np.mean(performance_list_SSP_peak_finder_v2_partial_sc))



#%% alternative optimization-based algorithm, based on Linglong dai's paper
# TTD's phases could be proportional to the RF or baseband subcarriers' frequencies
max_delay = (num_antenna_bs - 1) / (2 * fc) 
# max_delay = 1/eta/2 # 整数倍的情况下, 可以保证FFT based计算结果和非FFD based一模一样
grids = 256 
num_max_iter = 30

FFT_based = True
partial_fft = True

if FFT_based:
    # assert max_delay <= 1/eta
    grids = int(np.ceil(grids*(1/eta/max_delay)))
    # assert grids >= num_sc


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
        
        sum_rate = 0
        for n in range(num_sc):
            H_subcarrier_true = H_list_true[i,n]
            V = np.expand_dims(HBFs[n],axis=1)
            for k in range(num_user):
                H_k = H_subcarrier_true[:, :, k]  # NrxNt
                V_k = V[:, :, k]  # Ntx1
                signal_k = H_k.dot(V_k)
                signal_k_energy = signal_k.dot(np.transpose(np.conjugate(signal_k)))
                interference_k_energy = 0
                for j in range(num_user):
                    if j != k:
                        V_j = V[:, :, j]
                        interference_j = H_k.dot(V_j)
                        interference_k_energy = interference_k_energy + interference_j.dot(np.transpose(np.conjugate(interference_j)))
                SINR_k = signal_k_energy.dot(np.linalg.inv(interference_k_energy + sigma_2 * np.eye(num_antenna_ue)))
                rate_k = np.log2(np.linalg.det(np.eye(num_antenna_ue) + SINR_k))
                sum_rate = sum_rate + rate_k  

    performance_list_dll.append(np.real(sum_rate)/num_sc) 
    
print('Number of FFT points: %d'%grids)
print('Mean time of AM: %.1f ms' % (total_time * 1000 / num_samples))
print('Mean number of iterations of AM, with random initialization: %d'%(total_iters/num_samples))
print('Performance of AM, with random initialization: %.4f\n' %np.mean(performance_list_dll))
    

# AM with low-complexity extended SSP initialization
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
        
        sum_rate = 0
        for n in range(num_sc):
            H_subcarrier_true = H_list_true[i,n]
            V = np.expand_dims(HBFs[n],axis=1)
            for k in range(num_user):
                H_k = H_subcarrier_true[:, :, k]  # NrxNt
                V_k = V[:, :, k]  # Ntx1
                signal_k = H_k.dot(V_k)
                signal_k_energy = signal_k.dot(np.transpose(np.conjugate(signal_k)))
                interference_k_energy = 0
                for j in range(num_user):
                    if j != k:
                        V_j = V[:, :, j]
                        interference_j = H_k.dot(V_j)
                        interference_k_energy = interference_k_energy + interference_j.dot(np.transpose(np.conjugate(interference_j)))
                SINR_k = signal_k_energy.dot(np.linalg.inv(interference_k_energy + sigma_2 * np.eye(num_antenna_ue)))
                rate_k = np.log2(np.linalg.det(np.eye(num_antenna_ue) + SINR_k))
                sum_rate = sum_rate + rate_k  

    performance_list_dll.append(np.real(sum_rate)/num_sc) 
# print('FDB approximation mse of AM initialized by low-complexity extended SSP: %.5f'%(mse_with_iter[-1]/num_samples))    
print('Mean number of iterations of AM, initialized by LCE-SSP: %d'%(total_iters/num_samples))
print('Performance of AM, initialized by LCE-SSP: %.4f\n' %np.mean(performance_list_dll))

