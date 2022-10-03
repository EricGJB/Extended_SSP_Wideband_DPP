import numpy as np
import time
from scipy import io
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

np.random.seed(2022)

# system parameters
fc, W, tau_max = 100 * 1e9, 10 * 1e9, 20 * 1e-9
num_antenna_bs, num_antenna_ue, num_rf, num_sc, num_stream, num_TTD = 128, 4, 3, 8, 3, 8
eta = W / num_sc

Nc = 3

num_ps_per_TTD = num_antenna_bs//num_TTD

num_samples = 100
# saveflag = False

rou = 10 # dB
rou = 10**(rou/10) # linear SNR

# load testing channel samples
dataset = io.loadmat('./data/channel_%dclusters_%dscs_laplace.mat'%(Nc, num_sc))
H_list = dataset['H_list'][-num_samples:]
print(H_list.shape)  # (data_num, num_sc, num_antenna_bs, num_antenna_ue)
print('\n')

G_angle = 8*num_antenna_bs


#%% Baseline 1: fully digital beamforming, as upper bound
performance_list = np.zeros(num_samples)
FDBs_list = np.zeros((num_samples,num_sc,num_antenna_bs,num_stream))+1j*np.zeros((num_samples,num_sc,num_antenna_bs,num_stream))

# 0 is just svd
# 1 is waterfilling power allocation among data streams within each subcarrier
# 2 is further waterfilling power allocation among subcarriers
water_filling_level = 1

start = time.time()
for i in range(num_samples):
    if i % 500 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list = []

    if water_filling_level<2:
        for n in range(num_sc):
            # SVD decomposition, rule in numpy: A = U * Sigma * V
            H_subcarrier = H_list[i,n]
            U, Sigma, V = np.linalg.svd(np.transpose(np.conjugate(H_subcarrier)))
            fully_digital_beamformer = np.transpose(np.conjugate(V))
            # no need for normalization since already satisfied
            fully_digital_beamformer = fully_digital_beamformer[:,:num_stream]
            
            # 第一个元素相位归零
            # fully_digital_beamformer = fully_digital_beamformer[:,:num_stream]/(fully_digital_beamformer[0,:num_stream]/np.abs(fully_digital_beamformer[0,:num_stream]))

            # improve performance by further water-filling power allocation within each subcarrier
            if water_filling_level == 1:
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
            # print(np.linalg.norm(fully_digital_beamformer))
    
            # compute the rate, without combiner (won't affect the results)
            temp = np.transpose(np.conjugate(H_subcarrier)).dot(fully_digital_beamformer)
            temp = temp.dot(np.transpose(np.conjugate(temp)))
            rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
            rate_subcarrier_list.append(np.real(rate_subcarrier))

            FDBs_list[i, n] = fully_digital_beamformer


    else: # water filling among all subcarriers
        Sigma_list = np.zeros(num_sc * num_stream)
        fdb_dict = {}
        H_subcarrier_dict = {}    
        for n in range(num_sc):
            # SVD decomposition, rule: A = U * Sigma * V
            H_subcarrier = H_list[i, n]
            U, Sigma, V = np.linalg.svd(np.transpose(np.conjugate(H_subcarrier)))
            fully_digital_beamformer = np.transpose(np.conjugate(V))
            fully_digital_beamformer = fully_digital_beamformer[:, :num_stream]
            # per subcarrier energy normalization
            fully_digital_beamformer = fully_digital_beamformer / np.linalg.norm(fully_digital_beamformer) * np.sqrt(
                num_stream)
            
            Sigma_list[n * num_stream:(n + 1) * num_stream] = Sigma[:num_stream]
            fdb_dict[n] = fully_digital_beamformer
            H_subcarrier_dict[n] = H_subcarrier

        eq_noise_0 = num_stream/(rou*Sigma_list**2)
    
        # sort Sigma_list, in the descending order
        sorted_indexes = np.argsort(-Sigma_list)
        Sigma_list = Sigma_list[sorted_indexes]
        eq_noise = num_stream/(rou*Sigma_list**2)
        flag = 1 
        num_subchannels = num_stream*num_sc
        while flag:      
            water_level = (num_stream*num_sc+np.sum(eq_noise))/num_subchannels
            if water_level>np.max(eq_noise):
                flag=0
            else:
                eq_noise = eq_noise[:-1]
                num_subchannels = num_subchannels-1
        pa_vector = np.maximum(water_level-eq_noise_0, 0) #(1,num_sc*num_stream)
        
        for n in range(num_sc):
            fully_digital_beamformer = fdb_dict[n] * np.sqrt(np.expand_dims(pa_vector[n * num_stream:(n + 1) * num_stream], axis=0))
            # compute the rate, without combiner
            temp = np.transpose(np.conjugate(H_subcarrier_dict[n])).dot(fully_digital_beamformer)
            temp = temp.dot(np.transpose(np.conjugate(temp)))
            rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
            rate_subcarrier_list.append(np.real(rate_subcarrier))

            FDBs_list[i, n] = fully_digital_beamformer

    performance_list[i] = np.mean(rate_subcarrier_list)

end = time.time()
total_time = end - start

# print(FDBs_list.shape)  # (data_num, num_sc, num_antenna_bs, num_stream)

print('Mean time of FDB: %.1f ms'%(total_time/num_samples*1000))
print('Performance of the FDB upper bound: %.4f\n'%np.mean(performance_list))

# print('Performance of the FDB upper bound testing set: %.4f'%np.mean(performance_list[-int(0.1*num_samples):]))



#%% put common functions and values here
def dictionary_angle(N, G, sin_value):
    A = np.exp(-1j * np.pi * np.reshape(np.arange(N),(N,1)).dot(np.reshape(sin_value,(1,G))))
    return A
sin_value_sc0 = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)



#%% SSP with full TTDs
# construct dictionary matrices
A_list_full = []
for i in range(num_sc):
    sin_value_sci = sin_value_sc0*(1+i*eta/fc) # the first sc has accurate [-1,1] response vector 
    # sin_value_sci = sin_value_sc0 # frequency-independent measurement matrices
    # sin_value_sci = sin_value_sci/(1+(num_sc-1)*eta/fc) # the last
    # sin_value_sci = sin_value_sci/(1+(num_sc-1)/2*eta/fc) # the middle
    A_list_full.append(dictionary_angle(num_antenna_bs, G_angle, sin_value_sci))
A_list_full = np.array(A_list_full)



#%% plot responses of different subcarriers
## 针对256载波, 512 grid的情况
# A_common = dictionary_angle(num_antenna_bs, G_angle, sin_value_sc0)
# sample_index = 1
# FDBs = FDBs_list[sample_index]
# response_list_independent = np.zeros((num_sc,G_angle))
# response_list_dependent = np.zeros((num_sc,G_angle))
# for n in range(num_sc):
#     response_list_dependent[n] = np.linalg.norm(np.matrix(A_list_full[n]).H.dot(np.matrix(FDBs[n])),axis=-1)**2
#     response_list_independent[n] = np.linalg.norm(np.matrix(A_common).H.dot(np.matrix(FDBs[n])),axis=-1)**2

# fig = plt.figure()

# ax1 = fig.add_subplot(111)
# ax1.imshow(response_list_independent,cmap='gray_r')
# ax1.set_xlabel('Codeword Index')
# ax1.set_xlim([0,G_angle])
# ax1.set_xticks([0,64,128,192,256,320,384,448,512])
# ax1.set_ylim([0,num_sc])
# ax1.set_yticks([0,64,128,192,256])
# ax1.set_ylabel('Subcarrier index')

# average_responses_independent = np.abs(np.mean(response_list_independent,axis=0))
# ax2 = ax1.twinx()
# # ax2.set_ylim([0,40])
# # ax2.set_yticks([0,10,20,30,40])
# ax2.plot(average_responses_independent,'b--')
# ax2.set_ylabel('Average Projection')


# fig = plt.figure()

# ax1 = fig.add_subplot(111)
# ax1.imshow(response_list_dependent,cmap='gray_r')
# ax1.set_xlabel('Codeword Index')
# ax1.set_xlim([0,G_angle])
# ax1.set_xticks([0,64,128,192,256,320,384,448,512])
# ax1.set_ylim([0,num_sc])
# ax1.set_yticks([0,64,128,192,256])
# ax1.set_ylabel('Subcarrier index')

# average_responses_dependent = np.abs(np.mean(response_list_dependent,axis=0))
# ax2 = ax1.twinx()
# ax2.plot(average_responses_dependent,'r--')
# ax2.set_ylabel('Average Projection')


#%%
def SSP_wideband(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf):
    # 先考虑每个移相器都有一个TTD
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
    
    F_RF_list = []
    for i in range(num_rf):
        A_list_new = []
        responses = 0
        for n in range(num_sc):
            responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream)
        max_angle_index = np.argmax(responses)
        # print(max_angle_index)
        
        F_BB_list = []
        for n in range(num_sc):
            if i==0:
                F_RF_list.append(A_list[n,:,max_angle_index:max_angle_index+1])
            else:
                F_RF_list[n] = np.concatenate([F_RF_list[n],A_list[n,:,max_angle_index:max_angle_index+1]],axis=-1)
            A_list_new.append(np.concatenate([A_list[n,:,:max_angle_index],A_list[n,:,max_angle_index+1:]],axis=-1))
            F_BB_list.append(np.linalg.pinv(F_RF_list[n]).dot(FDBs[n]))
            residual[n] = FDBs[n] - F_RF_list[n].dot(F_BB_list[n]) 
            residual[n] = residual[n]/np.linalg.norm(residual[n])
            
        A_list = np.array(A_list_new)
        
    HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
    for n in range(num_sc):
        F_BB_list[n] = F_BB_list[n]*np.sqrt(num_stream)/np.linalg.norm(F_RF_list[n].dot(F_BB_list[n]))
        HBFs[n] = F_RF_list[n].dot(F_BB_list[n])   
    # print(np.linalg.norm(HBFs)**2)  
    return HBFs


performance_list_SSP_full = []
for i in range(num_samples):
    if i % 500 == 0:
        print('Testing sample %d' % i)
    FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
    # Optimization
    HBFs = SSP_wideband(FDBs, A_list_full, num_sc, num_antenna_bs, num_stream, num_rf)

    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier = H_list[i, n]
        temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_SSP_full.append(sum_rate/num_sc)

print('Performance of SSP upper bound with full TTDs: %.4f\n'%np.mean(performance_list_SSP_full))



#%% SSP with real achievable heuristic codebooks
# construct dictionary matrices
F_RF = dictionary_angle(num_antenna_bs, G_angle, sin_value_sc0)

PS_index_list = (2*np.arange(num_TTD)+1)*num_ps_per_TTD//2#-1
factor = PS_index_list/(2*fc)
T = np.expand_dims(sin_value_sc0,axis=-1).dot(np.expand_dims(factor,axis=0)) 

A_list_heuristic = []
for n in range(num_sc):
    f = n * eta # no fc if want to obtain heuristic solutions
    A_list_heuristic.append(np.transpose(F_RF) * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((1,num_ps_per_TTD))))
A_list_heuristic = np.transpose(A_list_heuristic,(0,2,1))

initial_mse = np.linalg.norm(A_list_heuristic-A_list_full)**2/np.product(A_list_full.shape)

def SSP_wideband_real_modified(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
    max_angle_indexes = []
    for i in range(num_rf):
        # compute the direction with largest average response energy
        responses = 0
        for n in range(num_sc):
            responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
        max_angle_index = np.argmax(responses)
        max_angle_indexes.append(max_angle_index)
        
        # if i==0:
        #     plt.figure()
        #     plt.plot(responses)
        #     print(np.argsort(responses)[-10:])        
        # print(max_angle_index)
            
        # update F_RF_n matrices with one vector added 
        if i==0:
            responses_first = responses
            F_RF_n_list = A_list[:,:,max_angle_index:max_angle_index+1]
        else:
            F_RF_n_list = np.concatenate([F_RF_n_list,A_list[:,:,max_angle_index:max_angle_index+1]],axis=-1)

        # update matrices in A_list with one vector removed
        # due to the "O" in OMP, the following line can be commented or not, with same results   
        # A_list = np.delete(A_list,max_angle_index,axis=-1)
        
        # obtain BB matrices
        HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
        for n in range(num_sc):
            F_RF_n = F_RF_n_list[n]
            F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
            residual[n] = FDBs[n] - F_RF_n.dot(F_BB_n) 
            residual[n] = residual[n]/np.linalg.norm(residual[n])
            if i==num_rf-1:
                HBF = F_RF_n.dot(F_BB_n)
                # normalization
                HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
                HBFs[n] = HBF
    
    return HBFs,max_angle_indexes,responses_first

# def post_process_index(indexes):
#     indexes_modified = []
#     for i in range(len(indexes)):
#         if i==0:
#             indexes_modified.append(indexes[i])
#         else:
#             step = 0
#             for j in range(i):
#                 if indexes[j]<=indexes[i]:
#                     step = step + 1
#             indexes_modified.append(indexes[i]+step)
#     return indexes_modified

# max_angle_indexes_list_true = np.zeros((num_samples,num_rf))
responses_list = np.zeros((num_samples,G_angle))

performance_list_SSP = []
mse_list_SSP = []
total_time = 0
for i in range(num_samples):
    if i % 500 == 0:
        print('Testing sample %d' % i)
    FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
    # Optimization
    start = time.time()
    HBFs,max_angle_indexes,responses = SSP_wideband_real_modified(FDBs, A_list_heuristic, num_sc, num_antenna_bs, num_stream, num_rf)
    end = time.time()
    total_time = total_time + end - start

    mse_SSP = np.linalg.norm(HBFs-FDBs)**2/np.product(HBFs.shape)
    mse_list_SSP.append(mse_SSP)
    
    # post processing indexes if the above SSP function is implemented with dynamically reduced A_list, otherwise no need
    # max_angle_indexes = post_process_index(max_angle_indexes)
    
    # max_angle_indexes_list_true[i] = max_angle_indexes
    responses_list[i] = responses
    
    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier = H_list[i, n]
        temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_SSP.append(sum_rate/num_sc)
    
print('Number of different delays in heuristic: %d'%len(np.unique(T)))  
print('Mean time of SSP heuristic: %.1f ms'%(total_time/num_samples*1000))
print('Performance of SSP heuristic: %.4f'%np.mean(performance_list_SSP))
print('FDB approximation mse of SSP heuristic: %.5f\n'%np.mean(mse_list_SSP))
# print('Performance of SSP testing set: %.4f'%np.mean(performance_list_SSP[-int(0.1*num_samples):]))

plt.figure()
plt.plot(np.abs(np.reshape(np.transpose(T),-1)))
# plt.figure()
# plt.imshow(np.angle(F_RF),cmap='gray_r')


# if saveflag:# save dataset
#     print(responses_list.shape)
#     print(max_angle_indexes_list_true.shape)
#     print('Data saved for DNN SU training\n')
#     io.savemat('./data/SSP_SU_dataset.mat',{'responses_list':responses_list,'max_angle_indexes_list':max_angle_indexes_list_true})


# double check the performance of indexes
# def SSP_wideband_real_modified(FDBs,A_list,num_sc,num_antenna_bs,num_stream,num_rf,indexes):
#     F_RF_n_list = A_list[:,:,indexes]
#     # obtain BB matrices
#     HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
#     for n in range(num_sc):
#         F_RF_n = F_RF_n_list[n]
#         F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
#         HBF = F_RF_n.dot(F_BB_n)
#         # normalization
#         HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
#         HBFs[n] = HBF
    
#     return HBFs

# performance_list_true = []
# true_indexes_list = max_angle_indexes_list_true.astype(np.int16)
# for i in range(num_samples):
#     if i % 500 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     HBFs_true = SSP_wideband_real_modified(FDBs, A_list, num_sc, num_antenna_bs, num_stream, num_rf, true_indexes_list[i])
#     sum_rate_true = 0
#     # Performance evaluation
#     for n in range(num_sc):
#         H_subcarrier = H_list[i, n]
#         temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs_true[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         sum_rate_true = sum_rate_true + np.real(rate_subcarrier)
#     performance_list_true.append(sum_rate_true/num_sc)

# print('Performance of SSP true: %.4f'%np.mean(performance_list_true))

# for i in range(num_samples):
#     if len(max_angle_indexes_list_true[i])!=num_rf:
#         break
# assert i==num_samples-1
# assert np.mean(performance_list_true)==np.mean(performance_list_SSP)



#%% use alternative optimization to obtain better codebooks
initial_A = F_RF
initial_T = np.transpose(T)

def dll(initial_A, initial_T, FDBs, num_sc, num_antennas_bs, num_TTD, num_rf, num_iter, grids, max_delay):
    object_list = np.zeros(num_iter+1)   
    HBFs_list = np.zeros((num_iter+1,num_sc,num_antennas_bs, num_rf),dtype=np.complex64)
    A_list = np.zeros((num_iter+1,num_antennas_bs, num_rf),dtype=np.complex64)
    T_list = np.zeros((num_iter+1,num_TTD, num_rf),dtype=np.complex64)
    
    # random initialization
    # initialize T
    T = np.random.uniform(0, 1, num_TTD * num_rf) * max_delay
    T = np.reshape(T, (num_TTD, num_rf))
    # initialize A
    A = np.random.uniform(-np.pi, np.pi, num_antennas_bs * num_rf)
    A = np.exp(1j * A)
    A = np.reshape(A, (num_antennas_bs, num_rf))
    
    # initialize with heuristic solutions 
    # A = initial_A
    # T = initial_T
    
    # initial evaluation
    HBFs = np.zeros((num_sc, num_antennas_bs, num_rf)) + 1j * np.zeros((num_sc, num_antennas_bs, num_rf))
    for n in range(num_sc):
        f = n * eta + fc # the initial evaluation uses the baseband model if initialized with heuristic codebooks
        HBF = A * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((num_ps_per_TTD, 1)))
        HBFs[n] = HBF
    objective = np.linalg.norm(HBFs - FDBs)**2/np.product(FDBs.shape)
    # print(objective)
    
    object_list[0] = objective
    HBFs_list[0] = HBFs
    A_list[0] = A
    T_list[0] = T

    # iterations
    for i in range(num_iter):
        # print('iteration %d'%i)
        ######################### update FRF
        term = 0
        for n in range(num_sc):
            f = n * eta + fc # 理论上加不加fc都行，不过实验发现加了性能更好
            Tm = np.exp(-1j * 2 * np.pi * f * T)
            # FBB_n = np.eye(num_rf,dtype=np.complex64)
            term = term + FDBs[n] * np.conjugate(np.kron(Tm, np.ones((num_ps_per_TTD, 1))))
        A = np.exp(1j * np.angle(term))

        ######################### update T
        TAU_list = []
        for n in range(num_sc):
            # FBB_n = np.eye(num_rf,dtype=np.complex64)
            Theta_m = FDBs[n] * np.conjugate(A)
            TAU = np.zeros((num_TTD, num_rf)) + 1j * np.zeros((num_TTD, num_rf))
            for l in range(num_TTD):
                for k in range(num_rf):
                    for p in range(num_ps_per_TTD):
                        TAU[l, k] = TAU[l, k] + Theta_m[l * num_ps_per_TTD + p, k]
            TAU_list.append(TAU)

        delay_list = np.arange(grids)/grids/eta
        stop_index = int(np.ceil(grids*max_delay*eta))
        # print(stop_index)
        # print(grids)
        factor_list = np.exp(-1j*2*np.pi*fc*delay_list)
        TAU_list = np.array(TAU_list)
        sequences = np.conjugate(TAU_list)
        
        for l in range(num_TTD):
            for k in range(num_rf):
                sequence = sequences[:,l,k]
                fft_responses = np.fft.fft(sequence,grids)*factor_list
                
                # fft_responses = fft_responses[:stop_index] 
                
                T[l,k] = delay_list[np.argmax(np.real(fft_responses))]
        
        HBFs = np.zeros((num_sc, num_antennas_bs, num_rf)) + 1j * np.zeros((num_sc, num_antennas_bs, num_rf))
        for n in range(num_sc):
            f = n * eta + fc
            HBF = A * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((num_ps_per_TTD, 1)))
            HBFs[n] = HBF
        objective = np.linalg.norm(HBFs - FDBs)**2/np.product(FDBs.shape)
        # print(objective)
        
        object_list[i+1] = objective
        HBFs_list[i+1] = HBFs
        A_list[i+1] = A
        T_list[i+1] = T

    return A_list, T_list, HBFs_list, np.array(object_list)

max_delay = (num_antenna_bs - 1) / (2 * fc) # *1.25
grids = 512*4

num_iter = 10 

assert max_delay <= 1/eta
grids = int(np.ceil(grids*(1/eta/max_delay)))
assert grids >= num_sc

A_optimized_iterations, T_optimized_iterations, A_list_optimized_iterations, object_list = dll(initial_A, initial_T, A_list_full, num_sc, num_antenna_bs, num_TTD, G_angle, num_iter, grids, max_delay)

optimal_iteration_index = np.where((object_list)==np.min(object_list))[0][0]
# if optimal_iteration_index == 0:
if np.min(object_list)>=initial_mse:
    print('No better codebook than the heuristic one is found')
    A_list_optimized = A_list_heuristic
    A_optimized = initial_A
    T_optimized = initial_T

else:
    print('Better codebook is found through optimization')
    A_list_optimized = A_list_optimized_iterations[optimal_iteration_index]
    A_optimized = A_optimized_iterations[optimal_iteration_index]
    T_optimized = T_optimized_iterations[optimal_iteration_index]

# run SSP with the optimized codebooks 
def SSP_wideband_real_modified(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
    max_angle_indexes = []
    for i in range(num_rf):
        # compute the direction with largest average response energy
        responses = 0
        for n in range(num_sc):
            responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
        max_angle_index = np.argmax(responses)
        max_angle_indexes.append(max_angle_index)
        
        # update F_RF_n matrices with one vector added 
        if i==0:
            responses_first = responses
            F_RF_n_list = A_list[:,:,max_angle_index:max_angle_index+1]
        else:
            F_RF_n_list = np.concatenate([F_RF_n_list,A_list[:,:,max_angle_index:max_angle_index+1]],axis=-1)
        
        # obtain BB matrices
        HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
        for n in range(num_sc):
            F_RF_n = F_RF_n_list[n]
            F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
            residual[n] = FDBs[n] - F_RF_n.dot(F_BB_n) 
            residual[n] = residual[n]/np.linalg.norm(residual[n])
            if i==num_rf-1:
                HBF = F_RF_n.dot(F_BB_n)
                # normalization
                HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
                HBFs[n] = HBF
    
    return HBFs,max_angle_indexes,responses_first

max_angle_indexes_list_optimized = np.zeros((num_samples,num_rf))
responses_list = np.zeros((num_samples,G_angle))

performance_list_SSP_optimized = []
mse_list_SSP_optimized = []
total_time = 0
for i in range(num_samples):
    if i % 500 == 0:
        print('Testing sample %d' % i)
    FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
    # Optimization
    start = time.time()
    HBFs,max_angle_indexes,responses = SSP_wideband_real_modified(FDBs, A_list_optimized, num_sc, num_antenna_bs, num_stream, num_rf)
    end = time.time()
    total_time = total_time + end - start
    
    mse_SSP_optimized = np.linalg.norm(HBFs-FDBs)**2/np.product(HBFs.shape)
    mse_list_SSP_optimized.append(mse_SSP_optimized)
    
    max_angle_indexes_list_optimized[i] = max_angle_indexes
    responses_list[i] = responses
    
    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier = H_list[i, n]
        temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_SSP_optimized.append(sum_rate/num_sc)
    
print('Number of different delays in optimized: %d'%len(np.unique(T_optimized)))
print('Performance of SSP optimized with larger delay: %.4f'%np.mean(performance_list_SSP_optimized))
print('FDB approximation mse of SSP optimized with larger delay range: %.5f\n'%np.mean(mse_list_SSP_optimized))

plt.figure()
plt.plot(object_list)
plt.figure()
plt.plot(np.abs(np.reshape(T_optimized,-1)))
# plt.figure()
# plt.imshow(np.angle(A_optimized),cmap='gray_r')

io.savemat('./data/A_list_optimized_%dscs_%dgrids.mat'%(num_sc,G_angle),{'A_list_optimized':A_list_optimized})


#%%SSP modified, with top K selection directly
def SSP_wideband_real_modified(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
 
    responses = 0
    for n in range(num_sc):
        responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
    max_angle_indexes = np.argsort(responses)[-num_rf:]
        
    F_RF_n_list = A_list[:,:,max_angle_indexes]
    
    # obtain BB matrices
    HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
    for n in range(num_sc):
        F_RF_n = F_RF_n_list[n]
        F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
        HBF = F_RF_n.dot(F_BB_n)
        # normalization
        HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
        HBFs[n] = HBF
    
    return HBFs,max_angle_indexes


max_angle_indexes_list_predicted = np.zeros((num_samples,num_rf))

performance_list_SSP_low_complexity = []
total_time = 0
for i in range(num_samples):
    if i % 500 == 0:
        print('Testing sample %d' % i)
    FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
    # Optimization
    start = time.time()
    HBFs,max_angle_indexes = SSP_wideband_real_modified(FDBs, A_list_optimized, num_sc, num_antenna_bs, num_stream, num_rf)
    max_angle_indexes_list_predicted[i] = max_angle_indexes
    
    end = time.time()
    total_time = total_time + end - start

    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier = H_list[i, n]
        temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_SSP_low_complexity.append(sum_rate/num_sc)
    
print('Performance of SSP low complexity: %.4f\n'%np.mean(performance_list_SSP_low_complexity))



#%% SSP modified, with peak finder algorithms
def peak_finder(responses,num_peaks):
    # zero padding
    responses = np.concatenate([np.zeros(1),responses,np.zeros(1)])
    # circular padding,性能有时变好有时变差，总体变差
    # 双边
    # responses = np.concatenate([responses[-1:],responses,responses[:1]])
    # 单边
    # responses = np.concatenate([responses[-1:],responses,np.zeros(1)])
    # responses = np.concatenate([np.zeros(1),responses,responses[:1]])
    peaks, peak_heights = find_peaks(responses, height=0)
    sorted_peaks = np.argsort(peak_heights['peak_heights'])
    assert len(sorted_peaks)>=num_peaks
    sorted_peaks_topK = sorted_peaks[-num_peaks:]
    peak_positions = peaks[sorted_peaks_topK]-1
    return peak_positions

def SSP_wideband_real_modified(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
 
    responses = 0
    for n in range(num_sc):
        responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
    max_angle_indexes = peak_finder(responses,num_rf)
        
    F_RF_n_list = A_list[:,:,max_angle_indexes]
    
    # obtain BB matrices
    HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
    for n in range(num_sc):
        F_RF_n = F_RF_n_list[n]
        F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
        HBF = F_RF_n.dot(F_BB_n)
        # normalization
        HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
        HBFs[n] = HBF
    
    return HBFs,max_angle_indexes


max_angle_indexes_list_peak = np.zeros((num_samples,num_rf))

performance_list_SSP_peak_finder = []
total_time = 0
for i in range(num_samples):
    if i % 500 == 0:
        print('Testing sample %d' % i)
    FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
    # Optimization
    start = time.time()
    HBFs,max_angle_indexes = SSP_wideband_real_modified(FDBs, A_list_optimized, num_sc, num_antenna_bs, num_stream, num_rf)
    max_angle_indexes_list_peak[i] = max_angle_indexes
    
    end = time.time()
    total_time = total_time + end - start
    # print(HBFs)

    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier = H_list[i, n]
        temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_SSP_peak_finder.append(sum_rate/num_sc)

error = 0
for i in range(num_samples):
    # if not (np.sort(max_angle_indexes_list_optimized[i]) == np.sort(max_angle_indexes_list_peak[i])).all():
        # error = error + 1
    for j in range(num_rf):
        if max_angle_indexes_list_peak[i,j] not in max_angle_indexes_list_optimized[i]:
            error = error + 1
accuracy = 1-error/num_samples/num_rf

# error = 0
# for i in range(int(num_samples*0.9),num_samples):
#     if not (np.sort(max_angle_indexes_list_optimized[i]) == np.sort(max_angle_indexes_list_peak[i])).all():
#         error = error + 1
# accuracy_testing = 1-error/int(num_samples*0.1)

print('Mean time of SSP peak_finder: %.1f ms'%(total_time/num_samples*1000))
print('Performance of SSP peak finder: %.4f'%np.mean(performance_list_SSP_peak_finder))
# print('Performance of SSP peak_finder testing set: %.4f'%np.mean(performance_list_SSP_peak_finder[-int(0.1*num_samples):]))
print('Support accuracy of peak_finder: %.3f\n'%accuracy)
# print('Support accuracy of peak_finder testing set: %.3f\n'%accuracy_testing)



#%% peak finder v2, with the hierarchical idea to achieve both high performance and low complexity
max_angle_indexes_list_peak_v2 = np.zeros((num_samples,num_rf))

performance_list_SSP_peak_finder_v2 = []
total_time = 0

down_sample_factor = 4
expand_factor = 2

def expand_indexes(max_angle_indexes,down_sample_factor,expand_factor,G_angle):
    max_angle_indexes_expanded = []
    max_angle_indexes = max_angle_indexes*down_sample_factor
    for max_angle_index in max_angle_indexes:
        max_angle_indexes_expanded.append(max_angle_index)
        for i in range(1,down_sample_factor*expand_factor):
            max_angle_indexes_expanded.append(max_angle_index+i)
            max_angle_indexes_expanded.append(max_angle_index-i)
    max_angle_indexes_expanded = np.array(max_angle_indexes_expanded)
    max_angle_indexes_expanded = np.maximum(max_angle_indexes_expanded,0)
    max_angle_indexes_expanded = np.minimum(max_angle_indexes_expanded,G_angle-1)
    # max_angle_indexes_expanded[np.where(max_angle_indexes_expanded<0)] = max_angle_indexes_expanded[np.where(max_angle_indexes_expanded<0)] + G_angle 
    # max_angle_indexes_expanded[np.where(max_angle_indexes_expanded>=G_angle)] = max_angle_indexes_expanded[np.where(max_angle_indexes_expanded>=G_angle)] - G_angle 
    return np.unique(max_angle_indexes_expanded)

def stage_1(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
 
    responses = 0
    for n in range(num_sc):
        responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
    max_angle_indexes = peak_finder(responses,num_rf)

    return max_angle_indexes

def peak_finder_v2(responses,num_peaks,max_angle_indexes_expanded):
    # zero padding
    responses_expanded = np.zeros(G_angle)
    responses_expanded[max_angle_indexes_expanded] = responses
    responses = np.concatenate([np.zeros(1),responses_expanded,np.zeros(1)])
    peaks, peak_heights = find_peaks(responses, height=0)
    sorted_peaks = np.argsort(peak_heights['peak_heights'])
    assert len(sorted_peaks)>=num_peaks
    sorted_peaks_topK = sorted_peaks[-num_peaks:]
    peak_positions = peaks[sorted_peaks_topK]-1
    return peak_positions

def SSP_wideband_real_modified(FDBs,A_list,A_list_full,num_sc,num_antenna_bs,num_stream, num_rf,max_angle_indexes_expanded):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
 
    responses = 0
    for n in range(num_sc):
        responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
    max_angle_indexes = peak_finder_v2(responses,num_rf,max_angle_indexes_expanded)
        
    F_RF_n_list = A_list_full[:,:,max_angle_indexes]
    
    # obtain BB matrices
    HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
    for n in range(num_sc):
        F_RF_n = F_RF_n_list[n]
        F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
        HBF = F_RF_n.dot(F_BB_n)
        # normalization
        HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
        HBFs[n] = HBF
    
    return HBFs,max_angle_indexes

for i in range(num_samples):
    if i % 500 == 0:
        print('Testing sample %d' % i)
    FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
    # Optimization
    start = time.time()
    
    # stage 1, coarse grids, find the position of peaks   
    max_angle_indexes = stage_1(FDBs, A_list_optimized[:,:,::down_sample_factor], num_sc, num_antenna_bs, num_stream, num_rf)
    max_angle_indexes_expanded = expand_indexes(max_angle_indexes,down_sample_factor,expand_factor,G_angle)
    HBFs,max_angle_indexes = SSP_wideband_real_modified(FDBs, A_list_optimized[:,:,max_angle_indexes_expanded], A_list_optimized, num_sc, num_antenna_bs, num_stream, num_rf,max_angle_indexes_expanded)
    max_angle_indexes_list_peak_v2[i] = max_angle_indexes
    
    end = time.time()
    total_time = total_time + end - start
    # print(HBFs)

    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier = H_list[i, n]
        temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_SSP_peak_finder_v2.append(sum_rate/num_sc)

error = 0
for i in range(num_samples):
    for j in range(num_rf):
        if max_angle_indexes_list_peak_v2[i,j] not in max_angle_indexes_list_optimized[i]:
            error = error + 1
accuracy = 1-error/num_samples/num_rf

# error = 0
# for i in range(int(num_samples*0.9),num_samples):
#     if not (np.sort(max_angle_indexes_list_optimized[i]) == np.sort(max_angle_indexes_list_peak_v2[i])).all():
#         error = error + 1
# accuracy_testing = 1-error/int(num_samples*0.1)

print('Mean time of SSP peak_finder v2: %.1f ms'%(total_time/num_samples*1000))
print('Performance of SSP peak finder v2: %.4f'%np.mean(performance_list_SSP_peak_finder_v2))
# print('Performance of SSP peak_finder v2 testing set: %.4f'%np.mean(performance_list_SSP_peak_finder_v2[-int(0.1*num_samples):]))
print('Support accuracy of peak_finder v2: %.3f\n'%accuracy)
# print('Support accuracy of peak_finder testing set v2: %.3f\n'%accuracy_testing)



#%% generalized sc version
num_sc_used = 8 # or 16
resolution = num_sc//num_sc_used

max_angle_indexes_list_peak_v2_partial_sc = np.zeros((num_samples,num_rf))

performance_list_SSP_peak_finder_v2_partial_sc = []
total_time = 0

down_sample_factor = 4
expand_factor = 2

def stage_2(FDBs,A_list,A_list_full,num_sc,num_antenna_bs,num_stream, num_rf,max_angle_indexes_expanded):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
 
    responses = 0
    for n in range(num_sc):
        responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
    max_angle_indexes = peak_finder_v2(responses,num_rf,max_angle_indexes_expanded)
        
    return max_angle_indexes


def stage_3(FDBs,A_list,max_angle_indexes,num_sc,num_antenna_bs,num_stream):
    F_RF_n_list = A_list[:,:,max_angle_indexes]
    # obtain BB matrices
    HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
    for n in range(num_sc):
        F_RF_n = F_RF_n_list[n]
        F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
        HBF = F_RF_n.dot(F_BB_n)
        # normalization
        HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
        HBFs[n] = HBF
    return HBFs


for i in range(num_samples):
    if i % 100 == 0:
        print('Testing sample %d' % i)
    FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
    # Optimization
    start = time.time()
    
    # stage 1, coarse grids, find the position of peaks   
    max_angle_indexes = stage_1(FDBs[::resolution], A_list_optimized[::resolution,:,::down_sample_factor], num_sc_used, num_antenna_bs, num_stream, num_rf)
    max_angle_indexes_expanded = expand_indexes(max_angle_indexes,down_sample_factor,expand_factor,G_angle)
    max_angle_indexes = stage_2(FDBs[::resolution], A_list_optimized[::resolution,:,max_angle_indexes_expanded], A_list_optimized[::resolution], num_sc_used, num_antenna_bs, num_stream, num_rf,max_angle_indexes_expanded)
    max_angle_indexes_list_peak_v2_partial_sc[i] = max_angle_indexes
    
    HBFs = stage_3(FDBs,A_list_optimized,max_angle_indexes,num_sc,num_antenna_bs,num_stream)
    
    end = time.time()
    total_time = total_time + end - start
    # print(HBFs)

    sum_rate = 0
    # Performance evaluation
    for n in range(num_sc):
        H_subcarrier = H_list[i, n]
        temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
        temp = temp.dot(np.transpose(np.conjugate(temp)))
        rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
        sum_rate = sum_rate + np.real(rate_subcarrier)

    performance_list_SSP_peak_finder_v2_partial_sc.append(sum_rate/num_sc)
    
print('Mean time of SSP peak_finder v2 partial sc: %.1f ms'%(total_time/num_samples*1000))
print('Performance of SSP peak finder v2 partial sc: %.4f\n'%np.mean(performance_list_SSP_peak_finder_v2_partial_sc))



#%% plot example of average projections
sample_index = 0
responses = responses_list[sample_index]
max_angle_indexes_optimized = max_angle_indexes_list_optimized[sample_index]
max_angle_indexes_predicted = max_angle_indexes_list_predicted[sample_index]
max_angle_indexes_peak = max_angle_indexes_list_peak[sample_index]
# max_angle_indexes_exhaustive = max_angle_indexes_list_exhaustive[sample_index]
shifts_optimized = 2000*np.ones(G_angle)
shifts_optimized[max_angle_indexes_optimized.astype(np.int16)] = shifts_optimized[max_angle_indexes_optimized.astype(np.int16)] - 2000
shifts_predicted = 2000*np.ones(G_angle)
shifts_predicted[max_angle_indexes_predicted.astype(np.int16)] = shifts_predicted[max_angle_indexes_predicted.astype(np.int16)] - 2000
shifts_peak = 2000*np.ones(G_angle)
shifts_peak[max_angle_indexes_peak.astype(np.int16)] = shifts_peak[max_angle_indexes_peak.astype(np.int16)] - 2000
# shifts_exhaustive = 2000*np.ones(G_angle)
# shifts_exhaustive[max_angle_indexes_exhaustive.astype(np.int16)] = shifts_exhaustive[max_angle_indexes_exhaustive.astype(np.int16)] - 2000
responses_optimized = responses + shifts_optimized
responses_predicted = responses + shifts_predicted
responses_peak = responses + shifts_peak
# responses_exhaustive = responses + shifts_exhaustive
plt.figure()
plt.plot(np.abs(responses),'k-',label=None)
plt.plot(np.abs(responses_optimized),'bo',markerfacecolor='white',label='Codewords Selected by Iterative SOMP')
plt.plot(np.abs(responses_predicted),'g+',label='Codewords Selected by Top-$N_{RF}$')
plt.plot(np.abs(responses_peak),'rx',label='Codewords Selected by Top-$N_{RF}$ Peaks')
# plt.plot(np.abs(responses_exhaustive),'gx')
# plt.xlim([0,G_angle])
# plt.xticks([0,32,64,96,128])
plt.ylim(0,1500)
plt.xlabel('Codeword Index')
plt.ylabel('Average Projection')
# plt.legend(['Average Projections','Supports of Peak Finder','Supports of Iterative SOMP'],loc='upper right')
plt.legend(loc='upper left')
plt.show()

print(max_angle_indexes_peak)
print(max_angle_indexes_predicted)
print(max_angle_indexes_optimized)


print(performance_list_SSP_peak_finder[sample_index])
print(performance_list_SSP_low_complexity[sample_index])
print(performance_list_SSP_optimized[sample_index])
print('\n')



#%% alternative optimization-based algorithm, based on Linglong dai's paper
# TTD's phases could be proportional to the RF or baseband subcarriers' frequencies
def dll(FDBs, num_sc, num_antennas_bs, num_stream, num_TTD, num_rf, num_iter, grids, max_delay, FFT_based):
    object_list = []
    # initialize T
    T = np.random.uniform(0, 1, num_TTD * num_rf) * max_delay
    T = np.reshape(T, (num_TTD, num_rf))
    # initialize A
    A = np.random.uniform(-np.pi, np.pi, num_antennas_bs * num_rf)
    A = np.exp(1j * A)
    A = np.reshape(A, (num_antennas_bs, num_rf))

    # iterations
    FBBs = np.zeros((num_sc, num_rf, num_stream)) + 1j * np.zeros((num_sc, num_rf, num_stream))
    HBFs_list = np.zeros((num_iter, num_sc, num_antennas_bs, num_stream)) + 1j * np.zeros((num_iter, num_sc, num_antennas_bs, num_stream))
    for i in range(num_iter):
        ############################# update FBBs
        for n in range(num_sc):
            # TODO: use which TTD phase model?
            f = n * eta + fc
            Tm = np.exp(-1j * 2 * np.pi * f * T)
            Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))
            Dm = np.linalg.pinv(Am).dot(FDBs[n])
            # normalization can be skipped during iterations according to the paper
            FBBs[n] = Dm

        ######################### update FRF
        term = 0
        for n in range(num_sc):
            f = n * eta + fc
            Tm = np.exp(-1j * 2 * np.pi * f * T)
            term = term + np.linalg.norm(FBBs[n]) ** 2 * (FDBs[n].dot(np.linalg.pinv(FBBs[n]))) * np.conjugate(np.kron(Tm, np.ones((num_ps_per_TTD, 1))))
        A = np.exp(1j * np.angle(term))

        ######################### update T
        # without TTDs, use 1 to avoid divide 0
        if num_TTD == 1:
            T = np.zeros((num_TTD, num_rf))
        else:
            TAU_list = []
            for n in range(num_sc):
                Theta_m = (FDBs[n].dot(np.linalg.pinv(FBBs[n]))) * np.conjugate(A)
                TAU = np.zeros((num_TTD, num_rf)) + 1j * np.zeros((num_TTD, num_rf))
                for l in range(num_TTD):
                    for k in range(num_rf):
                        for p in range(num_ps_per_TTD):
                            TAU[l, k] = TAU[l, k] + Theta_m[l * num_ps_per_TTD + p, k]
                TAU_list.append(TAU)

            # search based slow implementation\
            # TODO: fft based efficient implementation, how to map between fft and true delay
            if FFT_based:
                # x 1/eta instead of max_delay for FFT computation
                delay_list = np.arange(grids)/grids/eta
                stop_index = int(np.ceil(grids*max_delay*eta))
                # print(stop_index)
                # print(grids)
                factor_list = np.exp(-1j*2*np.pi*fc*delay_list)
                TAU_list = np.array(TAU_list)
                sequences = np.conjugate(TAU_list)*np.linalg.norm(FBBs,axis=(1,2),keepdims=True)**2
                for l in range(num_TTD):
                    for k in range(num_rf):
                        sequence = sequences[:,l,k]
                        # sequence_padded = np.zeros(grids)+1j*np.zeros(grids)
                        # sequence_padded[::grids//num_sc] = sequence
                        # sequence_padded[:num_sc] = sequence
                        # fft_responses = np.fft.fft(sequence_padded)*factor_list
                        fft_responses = np.fft.fft(sequence,grids)*factor_list
                        # only keep the beginning segment of fft responses due to the max_delay constraint
                        fft_responses = fft_responses[:stop_index]
                        T[l,k] = delay_list[np.argmax(np.real(fft_responses))]

            else:
                delay_list = np.arange(grids)/grids*max_delay
                for l in range(num_TTD):
                    for k in range(num_rf):
                        fft_responses = []
                        for delay in delay_list:
                            fft_response = 0
                            for n in range(num_sc):
                                f = n * eta + fc
                                fft_response = fft_response + np.conjugate(TAU_list[n][l, k]) * np.linalg.norm(
                                    FBBs[n]) ** 2 * np.exp(-1j * 2 * np.pi * f * delay)
                            fft_responses.append(fft_response)
                        T[l, k] = delay_list[np.argmax(np.real(fft_responses))]
                # plt.figure()
                # plt.plot(np.real(fft_responses))
                        
        HBFs = np.zeros((num_sc, num_antennas_bs, num_stream)) + 1j * np.zeros((num_sc, num_antennas_bs, num_stream))
        for n in range(num_sc):
            f = n * eta + fc
            HBF = (A * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((num_ps_per_TTD, 1)))).dot(FBBs[n])
            # normalization
            HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
            HBFs[n] = HBF
        object_list.append(np.linalg.norm(HBFs - FDBs) ** 2 / np.product(np.shape(HBFs)))
        HBFs_list[i] = HBFs

    return HBFs_list, np.array(object_list)


max_delay = (num_antenna_bs - 1) / (2 * fc) 
# max_delay = 1/eta

grids = 256 # or 128

num_iter = 20 # or 15
FFT_based = True

if FFT_based:
    assert max_delay <= 1/eta
    grids = int(np.ceil(grids*(1/eta/max_delay)))
    assert grids >= num_sc

performance_list_dll = np.zeros((num_samples, num_iter))
hybrid_beamformer_list = np.zeros((num_samples, num_sc, num_antenna_bs, num_stream)) + 1j * np.zeros((num_samples, num_sc, num_antenna_bs, num_stream))

total_time = 0
mse_with_iter = 0
for i in range(num_samples):
    if i % 500 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list_dll = np.zeros((num_iter, num_sc))
    # dll baseline
    start = time.time()
    HBFs_list, object_list_dll = dll(FDBs_list[i], num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_iter, grids, max_delay, FFT_based)
    end = time.time()
    total_time = total_time + (end - start)
    mse_with_iter = mse_with_iter + object_list_dll
    # best_iteration_index = np.where(object_list_dll==np.min(object_list_dll))[0][0]
    # hybrid_beamformer_list[i] = HBFs_list[best_iteration_index]
    hybrid_beamformer_list[i] = HBFs_list[-1]

    for it in range(num_iter):
        HBFs = HBFs_list[it]
        for n in range(num_sc):
            H_subcarrier = H_list[i, n]
            temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
            temp = temp.dot(np.transpose(np.conjugate(temp)))
            rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))

            rate_subcarrier_list_dll[it, n] = np.real(rate_subcarrier)

    performance_list_dll[i] = np.mean(rate_subcarrier_list_dll, axis=-1)
    
print('Mean time of alternative minimization: %.1f ms' % (total_time * 1000 / num_samples))
print('Performance of alternative minimization: %.4f\n' %np.mean(np.max(performance_list_dll, axis=-1)))
# print('Performance of the iterative algorithm testing set, mean: %.4f' %np.mean(np.max(performance_list_dll[-int(0.1*num_samples):], axis=-1)))


# print('Performance ratio of SSP: %.4f' % (np.mean(performance_list_SSP) / np.mean(performance_list)))
# print('Performance ratio of SSP peak finder: %.4f' % (np.mean(performance_list_SSP_peak_finder) / np.mean(performance_list)))
# print('Performance ratio of SSP peak finder v2: %.4f' % (np.mean(performance_list_SSP_peak_finder_v2) / np.mean(performance_list)))
# print('Performance ratio of iterative: %.4f\n' % (np.mean(np.max(performance_list_dll, axis=-1)) / np.mean(performance_list)))

# print('Performance ratio of SSP low complexity: %.4f' % (np.mean(performance_list_SSP_low_complexity) / np.mean(performance_list)))
# print('Performance ratio of SSP exhaustive: %.4f' % (np.mean(np.max(performance_list_SSP_exhaustive, axis=-1)) / np.mean(performance_list)))



#%% alternative optimization with larger delay
def dll(FDBs, num_sc, num_antennas_bs, num_stream, num_TTD, num_rf, num_iter, grids, max_delay, FFT_based):
    object_list = []
    # initialize T
    T = np.random.uniform(0, 1, num_TTD * num_rf) * max_delay
    T = np.reshape(T, (num_TTD, num_rf))
    # initialize A
    A = np.random.uniform(-np.pi, np.pi, num_antennas_bs * num_rf)
    A = np.exp(1j * A)
    A = np.reshape(A, (num_antennas_bs, num_rf))

    # iterations
    FBBs = np.zeros((num_sc, num_rf, num_stream)) + 1j * np.zeros((num_sc, num_rf, num_stream))
    HBFs_list = np.zeros((num_iter, num_sc, num_antennas_bs, num_stream)) + 1j * np.zeros((num_iter, num_sc, num_antennas_bs, num_stream))
    for i in range(num_iter):
        ############################# update FBBs
        for n in range(num_sc):
            # TODO: use which TTD phase model?
            f = n * eta + fc
            Tm = np.exp(-1j * 2 * np.pi * f * T)
            Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))
            Dm = np.linalg.pinv(Am).dot(FDBs[n])
            # normalization can be skipped during iterations according to the paper
            FBBs[n] = Dm

        ######################### update FRF
        term = 0
        for n in range(num_sc):
            f = n * eta + fc
            Tm = np.exp(-1j * 2 * np.pi * f * T)
            term = term + np.linalg.norm(FBBs[n]) ** 2 * (FDBs[n].dot(np.linalg.pinv(FBBs[n]))) * np.conjugate(np.kron(Tm, np.ones((num_ps_per_TTD, 1))))
        A = np.exp(1j * np.angle(term))

        ######################### update T
        # without TTDs, use 1 to avoid divide 0
        if num_TTD == 1:
            T = np.zeros((num_TTD, num_rf))
        else:
            TAU_list = []
            for n in range(num_sc):
                Theta_m = (FDBs[n].dot(np.linalg.pinv(FBBs[n]))) * np.conjugate(A)
                TAU = np.zeros((num_TTD, num_rf)) + 1j * np.zeros((num_TTD, num_rf))
                for l in range(num_TTD):
                    for k in range(num_rf):
                        for p in range(num_ps_per_TTD):
                            TAU[l, k] = TAU[l, k] + Theta_m[l * num_ps_per_TTD + p, k]
                TAU_list.append(TAU)

            # search based slow implementation\
            # TODO: fft based efficient implementation, how to map between fft and true delay
            if FFT_based:
                # x 1/eta instead of max_delay for FFT computation
                delay_list = np.arange(grids)/grids/eta
                stop_index = int(np.ceil(grids*max_delay*eta))
                # print(stop_index)
                # print(grids)
                factor_list = np.exp(-1j*2*np.pi*fc*delay_list)
                TAU_list = np.array(TAU_list)
                sequences = np.conjugate(TAU_list)*np.linalg.norm(FBBs,axis=(1,2),keepdims=True)**2
                for l in range(num_TTD):
                    for k in range(num_rf):
                        sequence = sequences[:,l,k]
                        # sequence_padded = np.zeros(grids)+1j*np.zeros(grids)
                        # sequence_padded[::grids//num_sc] = sequence
                        # sequence_padded[:num_sc] = sequence
                        # fft_responses = np.fft.fft(sequence_padded)*factor_list
                        fft_responses = np.fft.fft(sequence,grids)*factor_list
                        
                        # only keep the beginning segment of fft responses due to the max_delay constraint
                        # fft_responses = fft_responses[:stop_index]
                        
                        T[l,k] = delay_list[np.argmax(np.real(fft_responses))]

            else:
                delay_list = np.arange(grids)/grids*max_delay
                for l in range(num_TTD):
                    for k in range(num_rf):
                        fft_responses = []
                        for delay in delay_list:
                            fft_response = 0
                            for n in range(num_sc):
                                f = n * eta + fc
                                fft_response = fft_response + np.conjugate(TAU_list[n][l, k]) * np.linalg.norm(
                                    FBBs[n]) ** 2 * np.exp(-1j * 2 * np.pi * f * delay)
                            fft_responses.append(fft_response)
                        T[l, k] = delay_list[np.argmax(np.real(fft_responses))]
                # plt.figure()
                # plt.plot(np.real(fft_responses))
                        
        HBFs = np.zeros((num_sc, num_antennas_bs, num_stream)) + 1j * np.zeros((num_sc, num_antennas_bs, num_stream))
        for n in range(num_sc):
            f = n * eta + fc
            HBF = (A * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((num_ps_per_TTD, 1)))).dot(FBBs[n])
            # normalization
            HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
            HBFs[n] = HBF
        object_list.append(np.linalg.norm(HBFs - FDBs) ** 2 / np.product(np.shape(HBFs)))
        HBFs_list[i] = HBFs

    return HBFs_list, np.array(object_list)


max_delay = (num_antenna_bs - 1) / (2 * fc) # *1.25
# max_delay = 1/eta

grids = 256 # or 128

num_iter = 20 # or 15
FFT_based = True

if FFT_based:
    assert max_delay <= 1/eta
    grids = int(np.ceil(grids*(1/eta/max_delay)))
    assert grids >= num_sc

performance_list_dll_larger_delay = np.zeros((num_samples, num_iter))
hybrid_beamformer_list = np.zeros((num_samples, num_sc, num_antenna_bs, num_stream)) + 1j * np.zeros((num_samples, num_sc, num_antenna_bs, num_stream))

total_time = 0
mse_with_iter = 0
for i in range(num_samples):
    if i % 500 == 0:
        print('Testing sample %d' % i)
    rate_subcarrier_list_dll = np.zeros((num_iter, num_sc))
    # dll baseline
    start = time.time()
    HBFs_list, object_list_dll = dll(FDBs_list[i], num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_iter, grids, max_delay, FFT_based)
    end = time.time()
    total_time = total_time + (end - start)
    mse_with_iter = mse_with_iter + object_list_dll
    # best_iteration_index = np.where(object_list_dll==np.min(object_list_dll))[0][0]
    # hybrid_beamformer_list[i] = HBFs_list[best_iteration_index]
    hybrid_beamformer_list[i] = HBFs_list[-1]

    for it in range(num_iter):
        HBFs = HBFs_list[it]
        for n in range(num_sc):
            H_subcarrier = H_list[i, n]
            temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
            temp = temp.dot(np.transpose(np.conjugate(temp)))
            rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))

            rate_subcarrier_list_dll[it, n] = np.real(rate_subcarrier)

    performance_list_dll_larger_delay[i] = np.mean(rate_subcarrier_list_dll, axis=-1)

print('Performance of alternative minimization with larger delay range: %.4f' %np.mean(np.max(performance_list_dll_larger_delay, axis=-1)))
print('FDB approximation mse of alternative minimization with larger delay range: %.5f'%(mse_with_iter[-1]/num_samples))

plt.figure()
plt.plot(mse_with_iter/num_samples)
plt.plot(np.mean(mse_list_SSP_optimized)*np.ones(num_iter))
plt.legend(['AM','SSP'])
plt.ylabel('FDB approximation mse')


#%% SSP with post processing to obtain real FBB, FRF and T matrices
# construct dictionary matrices
# def dictionary_angle(N, G, sin_value):
#     A = np.exp(-1j * np.pi * np.reshape(np.arange(N),(N,1)).dot(np.reshape(sin_value,(1,G))))
#     return A
# A_list = []
# sin_value_sc0 = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)
# # sin_value_sc0 = np.linspace(-0.65 + 1 / G_angle, 0.65 - 1 / G_angle, G_angle)
# # sin_value_sc0 = np.concatenate([np.linspace(-1,-0.7,G_angle//3+1), np.linspace(-0.4,0.4,G_angle//3) ,np.linspace(0.7,1,G_angle//3+1)])
# for i in range(num_sc):
#     sin_value_sci = sin_value_sc0*(1+i*eta/fc)
#     A_list.append(dictionary_angle(num_antenna_bs, G_angle, sin_value_sci))

# A_list = np.array(A_list)

# def SSP_wideband_real(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf,sin_value_sc0):
#     residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
#     PS_index_list = (2*np.arange(num_TTD)+1)*num_ps_per_TTD//2-1
#     factor = PS_index_list/(2*fc)
    
#     for i in range(num_rf):
#         # compute the direction with largest average response energy
#         responses = 0
#         for n in range(num_sc):
#             responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream)
#         max_angle_index = np.argmax(responses)
#         # print(max_angle_index)
        
#         # obtain the rf vector of this chain 
#         f_RF = A_list[0,:,max_angle_index:max_angle_index+1]
        
#         # obtain the t vector of this chain
#         t = sin_value_sc0[max_angle_index]*factor # tmax = num_antenna_bs/(2*fc), the same as in iterative optimization 
#         t = np.expand_dims(t,axis=-1)
        
#         # update RF and T matrices with one vector added 
#         if i==0:
#             F_RF = f_RF
#             T = t
#         else:
#             F_RF = np.concatenate([F_RF,f_RF],axis=-1)
#             T = np.concatenate([T,t],axis=-1)
            
#         # update matrices in A_list with one vector removed
#         A_list = np.concatenate([A_list[:,:,:max_angle_index],A_list[:,:,max_angle_index+1:]],axis=-1)

#         # obtain BB matrices
#         HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
#         for n in range(num_sc):
#             f = n * eta # + fc, fc的影响可以通过后处理RF矩阵等效地消除
#             F_RF_n = F_RF * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((num_ps_per_TTD, 1)))
#             F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
#             residual[n] = FDBs[n] - F_RF_n.dot(F_BB_n) 
#             residual[n] = residual[n]/np.linalg.norm(residual[n])
#             if i==num_rf-1:
#                 HBF = F_RF_n.dot(F_BB_n)
#                 # normalization
#                 HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
#                 HBFs[n] = HBF
                
#     # print(np.linalg.norm(HBFs)**2)
#     # print(T)
    
#     return HBFs,F_RF,T


# performance_list_SSP = []
# total_time = 0
# for i in range(num_samples):
#     if i % 500 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     start = time.time()
#     HBFs,F_RF,T = SSP_wideband_real(FDBs, A_list, num_sc, num_antenna_bs, num_stream, num_rf,sin_value_sc0)
#     end = time.time()
#     total_time = total_time + end - start
#     # print(HBFs)

#     sum_rate = 0
#     # Performance evaluation
#     for n in range(num_sc):
#         H_subcarrier = H_list[i, n]
#         temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         sum_rate = sum_rate + np.real(rate_subcarrier)

#     performance_list_SSP.append(sum_rate/num_sc)

# print('Performance of SSP: %.4f'%np.mean(performance_list_SSP))
# print('Mean time of SSP: %.1f ms\n'%(total_time/num_samples*1000))


#%% low complexity v2, with orthogonal atoms
# to make sure the matrix is DFT
# assert G_angle == num_antenna_bs
# assert num_sc == 1

# def dictionary_angle(N, G, sin_value):
#     A = np.exp(-1j * np.pi * np.reshape(np.arange(N),(N,1)).dot(np.reshape(sin_value,(1,G))))
#     return A
# sin_value_sc0 = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)
# F_RF = dictionary_angle(num_antenna_bs, G_angle, sin_value_sc0)

# PS_index_list = (2*np.arange(num_TTD)+1)*num_ps_per_TTD//2-1
# factor = PS_index_list/(2*fc)
# T = np.expand_dims(sin_value_sc0,axis=-1).dot(np.expand_dims(factor,axis=0)) 

# A_list = []
# for n in range(num_sc):
#     f = n * eta 
#     A_list.append(np.transpose(F_RF) * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((1,num_ps_per_TTD))))
# A_list = np.transpose(A_list,(0,2,1))

# def SSP_wideband_real_modified(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf):
#     residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
 
#     Phi = np.matrix(A_list[0]).H.dot(np.matrix(residual[0]))   
 
#     responses = np.linalg.norm(Phi,axis=-1)**2 # (G_angle_left, num_stream) inside norm
    
#     max_angle_indexes = np.argsort(responses)[-num_rf:]
        
#     F_RF_n_list = A_list[:,:,max_angle_indexes]
    
#     # obtain BB matrices
#     HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
#     for n in range(num_sc):
#         F_RF_n = F_RF_n_list[n]
#         F_BB_n = Phi[max_angle_indexes]
#         HBF = F_RF_n.dot(F_BB_n)
#         # normalization
#         HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
#         HBFs[n] = HBF
    
#     return HBFs,max_angle_indexes


# max_angle_indexes_list_predicted = np.zeros((num_samples,num_rf))

# performance_list_SSP_low_complexity = []
# total_time = 0
# for i in range(num_samples):
#     if i % 500 == 0:
#         print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     start = time.time()
#     HBFs,max_angle_indexes = SSP_wideband_real_modified(FDBs, A_list, num_sc, num_antenna_bs, num_stream, num_rf)
#     max_angle_indexes_list_predicted[i] = max_angle_indexes
    
#     end = time.time()
#     total_time = total_time + end - start
#     # print(HBFs)

#     sum_rate = 0
#     # Performance evaluation
#     for n in range(num_sc):
#         H_subcarrier = H_list[i, n]
#         temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
#         temp = temp.dot(np.transpose(np.conjugate(temp)))
#         rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#         sum_rate = sum_rate + np.real(rate_subcarrier)

#     performance_list_SSP_low_complexity.append(sum_rate/num_sc)

# error = 0
# for i in range(num_samples):
#     # if (np.sort(max_angle_indexes_list_true[i]) != np.sort(max_angle_indexes_list_predicted[i])).all():
#     if not (np.sort(max_angle_indexes_list_true[i]) == np.sort(max_angle_indexes_list_predicted[i])).all():
#         error = error + 1
# accuracy = 1-error/num_samples

# error = 0
# for i in range(int(num_samples*0.9),num_samples):
#     if not (np.sort(max_angle_indexes_list_true[i]) == np.sort(max_angle_indexes_list_predicted[i])).all():
#         error = error + 1
# accuracy_testing = 1-error/int(num_samples*0.1)

# print('Performance of SSP low complexity: %.4f'%np.mean(performance_list_SSP_low_complexity))
# print('Performance of SSP low complexity testing set: %.4f'%np.mean(performance_list_SSP_low_complexity[-int(0.1*num_samples):]))
# print('Mean time of SSP low complexity: %.1f ms'%(total_time/num_samples*1000))
# print('Support accuracy of low complexity: %.3f'%accuracy)
# print('Support accuracy of low complexity testing set: %.3f\n'%accuracy_testing)



#%% obtain the support label with thresholded reduced exhaustive search
# def dictionary_angle(N, G, sin_value):
#     A = np.exp(-1j * np.pi * np.reshape(np.arange(N),(N,1)).dot(np.reshape(sin_value,(1,G))))
#     return A
# sin_value_sc0 = np.linspace(-1 + 1 / G_angle, 1 - 1 / G_angle, G_angle)
# F_RF = dictionary_angle(num_antenna_bs, G_angle, sin_value_sc0)

# PS_index_list = (2*np.arange(num_TTD)+1)*num_ps_per_TTD//2-1
# factor = PS_index_list/(2*fc)
# T = np.expand_dims(sin_value_sc0,axis=-1).dot(np.expand_dims(factor,axis=0)) 

# A_list = []
# for n in range(num_sc):
#     f = n * eta 
#     A_list.append(np.transpose(F_RF) * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((1,num_ps_per_TTD))))
# A_list = np.transpose(A_list,(0,2,1))

# from itertools import combinations

# def SSP_wideband_real_modified(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf, threshold):
#     residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)

#     responses = 0
#     for n in range(num_sc):
#         responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
#     angle_indexes_set = np.where(responses>=threshold)
    
#     print(len(angle_indexes_set[0]))
    
#     combins = [c for c in combinations(angle_indexes_set[0], num_rf)]
#     # print(len(combins))
#     max_rate = 0
    
#     for angle_indexes in combins:
#         F_RF_n_list = A_list[:,:,angle_indexes]      
#         # obtain BB matrices
#         HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
#         for n in range(num_sc):
#             F_RF_n = F_RF_n_list[n]
#             F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
#             HBF = F_RF_n.dot(F_BB_n)
#             # normalization
#             HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
#             HBFs[n] = HBF
        
#         # evaluate performance
#         sum_rate = 0
#         # Performance evaluation
#         for n in range(num_sc):
#             H_subcarrier = H_list[i, n]
#             temp = np.transpose(np.conjugate(H_subcarrier)).dot(HBFs[n])
#             temp = temp.dot(np.transpose(np.conjugate(temp)))
#             rate_subcarrier = np.log2(np.linalg.det(np.eye(num_antenna_ue) + rou / num_stream * temp))
#             sum_rate = sum_rate + np.real(rate_subcarrier)
#         if sum_rate > max_rate:
#             max_rate = sum_rate
#             max_angle_indexes = angle_indexes
    
#     return max_angle_indexes,max_rate/num_sc,responses

# threshold = 150 

# max_angle_indexes_list_exhaustive = np.zeros((num_samples,num_rf))
# responses_list = np.zeros((num_samples,G_angle))
# performance_list_SSP_exhaustive = []

# total_time = 0
# for i in range(num_samples):
#     # if i % 10 == 0:
#     print('Testing sample %d' % i)
#     FDBs = FDBs_list[i] #(num_sc*num_stream, num_antenna_bs)
#     # Optimization
#     start = time.time()
#     max_angle_indexes,max_rate_average,responses = SSP_wideband_real_modified(FDBs, A_list, num_sc, num_antenna_bs, num_stream, num_rf, threshold)
#     end = time.time()
#     total_time = total_time + end - start

#     max_angle_indexes_list_exhaustive[i] = max_angle_indexes
#     responses_list[i] = responses
#     performance_list_SSP_exhaustive.append(max_rate_average)

# print('Performance of SSP exhaustive: %.4f'%np.mean(performance_list_SSP_exhaustive))
# print('Performance of SSP exhaustive testing set: %.4f'%np.mean(performance_list_SSP_exhaustive[-int(0.1*num_samples):]))
# print('Mean time of SSP exhaustive: %.1f ms'%(total_time/num_samples*1000))

# if saveflag:# save dataset
#     print(responses_list.shape)
#     print(max_angle_indexes_list_exhaustive.shape)
#     print('Data saved for DNN SU training\n')
#     io.savemat('./data/SSP_SU_dataset.mat',{'responses_list':responses_list,'max_angle_indexes_list_exhaustive':max_angle_indexes_list_exhaustive})



#%%
# plt.figure()
# plt.plot(np.arange(num_iter),np.mean(performance_list_dll, axis=0))
# plt.plot(np.arange(num_iter),np.ones(num_iter)*np.mean(performance_list_SSP_modified))
# plt.plot(np.arange(num_iter),np.ones(num_iter)*np.mean(performance_list))
# plt.xticks(np.linspace(0,num_iter,4))
# plt.xlim(0,num_iter)
# plt.ylim(15,22)
# plt.xlabel('Number of iterations')
# plt.ylabel('Average Rate')
# plt.legend(['Opt-based','Extended SSP','Fully digital'],loc='lower right')


# # hist and smooth CDF curve
# plt.figure()
# values = plt.hist(np.max(performance_list_dll, axis=-1),bins=1000,cumulative=True,density=True)
# plt.figure()
# values_SSP = plt.hist(np.squeeze(performance_list_SSP_modified),bins=1000,cumulative=True,density=True)
# plt.figure()
# values_FDB = plt.hist(performance_list,bins=1000,cumulative=True,density=True)

# plt.figure()
# plt.plot(values[1],np.concatenate([np.zeros(1),values[0]],axis=0))
# plt.plot(values_SSP[1],np.concatenate([np.zeros(1),values_SSP[0]],axis=0))
# plt.plot(values_FDB[1],np.concatenate([np.zeros(1),values_FDB[0]],axis=0))
# plt.xlabel('Average Rate')
# plt.ylabel('CDF')
# plt.legend(['Opt-based','Extended SSP','Fully digital'],loc='lower right')


