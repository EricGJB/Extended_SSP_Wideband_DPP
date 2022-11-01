import numpy as np
np.random.seed(2022)
import time
from scipy.signal import find_peaks


def Extended_SSP_wideband_H(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf):
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
        
        for n in range(num_sc):
            F_RF_n = F_RF_n_list[n]
            F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
            residual[n] = FDBs[n] - F_RF_n.dot(F_BB_n) 
    
    return max_angle_indexes,responses_first



def Extended_SSP_wideband(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
    max_angle_indexes = []
    
    HBFs = np.zeros((num_sc,num_antenna_bs,num_stream),dtype=np.complex64)
    # F_BBs = np.zeros((num_sc,num_rf,num_stream),dtype=np.complex64) 
        
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
        
        for n in range(num_sc):
            F_RF_n = F_RF_n_list[n]
            F_BB_n = np.linalg.pinv(F_RF_n).dot(FDBs[n])
            residual[n] = FDBs[n] - F_RF_n.dot(F_BB_n) 
            residual[n] = residual[n]/np.linalg.norm(residual[n]) # this step seems to be unnecessary
            if i==num_rf-1:
                HBF = F_RF_n.dot(F_BB_n)
                # normalization
                scaler = 1 / np.linalg.norm(HBF) * np.sqrt(num_stream)
                HBF = HBF * scaler 
                
                HBFs[n] = HBF
                # F_BB_n = F_BB_n * scaler
                # F_BBs[n] = F_BB_n
    
    return HBFs,max_angle_indexes,responses_first


## ideal frequency-dependent dictionary matrices, with narrow beamwidth ULA response form 
def dictionary_angle(N, G, sin_value):
    A = np.exp(-1j * np.pi * np.reshape(np.arange(N),(N,1)).dot(np.reshape(sin_value,(1,G))))
    return A


def peak_finder(responses,num_peaks):
    # zero padding
    responses = np.concatenate([np.zeros(1),responses,np.zeros(1)])
    peaks, peak_heights = find_peaks(responses, height=0)
    sorted_peaks = np.argsort(peak_heights['peak_heights'])
    assert len(sorted_peaks)>=num_peaks
    sorted_peaks_topK = sorted_peaks[-num_peaks:]
    peak_positions = peaks[sorted_peaks_topK]-1
    return peak_positions


def TopK(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf, peak_finding):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
 
    responses = 0
    for n in range(num_sc):
        responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
    if peak_finding:
        max_angle_indexes = peak_finder(responses,num_rf)
    else:
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
    return np.unique(max_angle_indexes_expanded)


def stage_1(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
 
    responses = 0
    for n in range(num_sc):
        responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
    max_angle_indexes = peak_finder(responses,num_rf)

    return max_angle_indexes


def peak_finder_v2(responses,num_peaks,max_angle_indexes_expanded,G_angle):
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


def Hierarchical(FDBs,A_list,A_list_ideal,num_sc,num_antenna_bs,num_stream, num_rf,max_angle_indexes_expanded):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
 
    responses = 0
    for n in range(num_sc):
        responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
    max_angle_indexes = peak_finder_v2(responses,num_rf,max_angle_indexes_expanded)
        
    F_RF_n_list = A_list_ideal[:,:,max_angle_indexes]
    
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


def stage_2(FDBs,A_list,num_sc,num_antenna_bs,num_stream, num_rf,max_angle_indexes_expanded,G_angle):
    residual = np.copy(FDBs) #(num_sc, num_antenna_bs, num_stream)
 
    responses = 0
    for n in range(num_sc):
        responses = responses + np.linalg.norm(np.matrix(A_list[n]).H.dot(np.matrix(residual[n])),axis=-1)**2 # (G_angle_left, num_stream) inside norm
    max_angle_indexes = peak_finder_v2(responses,num_rf,max_angle_indexes_expanded,G_angle)
        
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


def AM(early_stop, epsilon, initial_A, initial_T, FDBs, fc, num_ps_per_TTD, eta, num_sc, num_antenna_bs, num_stream, num_TTD, num_rf, num_max_iter, grids, max_delay, FFT_based, partial_fft):
    HBFs_list = []
    A_list = []
    T_list = []
    object_list = []
    
    # must use .copy since T is updated element-wisely 
    A = np.copy(initial_A)
    T = np.copy(initial_T) 

    A_list.append(A)
    T_list.append(T)
    
    FBBs = np.zeros((num_sc, num_rf, num_stream)) + 1j * np.zeros((num_sc, num_rf, num_stream))
    HBFs = np.zeros((num_sc, num_antenna_bs, num_stream)) + 1j * np.zeros((num_sc, num_antenna_bs, num_stream))
    for n in range(num_sc):
        f = fc + (n-(num_sc-1)/2)*eta
        Tm = np.exp(-1j * 2 * np.pi * f * T)
        Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))
        Dm = np.linalg.pinv(Am).dot(FDBs[n])
        HBF = Am.dot(Dm)
        # normalization
        HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
        HBFs[n] = HBF
    object_list.append(np.linalg.norm(HBFs - FDBs) ** 2 / np.product(np.shape(HBFs)))
    HBFs_list.append(HBFs)
    
    time_elapse = 0
    
    for i in range(num_max_iter):
        start = time.time()
        
        ############################# update FBBs
        for n in range(num_sc):
            f = fc + (n-(num_sc-1)/2)*eta
            Tm = np.exp(-1j * 2 * np.pi * f * T)
            Am = A * np.kron(Tm, np.ones((num_ps_per_TTD, 1)))
            Dm = np.linalg.pinv(Am).dot(FDBs[n])
            # normalization can be skipped during iterations according to the paper
            FBBs[n] = Dm

        ######################### update FRF
        term = 0
        Theta_list = np.zeros((num_sc,num_antenna_bs,num_rf),dtype=np.complex64)
        for n in range(num_sc):
            f = fc + (n-(num_sc-1)/2)*eta
            Tm = np.exp(-1j * 2 * np.pi * f * T)
            Theta_n = FDBs[n].dot(np.linalg.pinv(FBBs[n]))
            term = term + np.linalg.norm(FBBs[n]) ** 2 * Theta_n * np.conjugate(np.kron(Tm, np.ones((num_ps_per_TTD, 1))))
            Theta_list[n] = Theta_n
        A = np.exp(1j * np.angle(term))

        ######################### update T
        T = np.zeros((num_TTD, num_rf))
        
        TAU_list = []
        for n in range(num_sc):
            Theta_m = Theta_list[n] * np.conjugate(A)
            TAU = np.zeros((num_TTD, num_rf)) + 1j * np.zeros((num_TTD, num_rf))
            for l in range(num_TTD):
                for k in range(num_rf):
                    for p in range(num_ps_per_TTD):
                        TAU[l, k] = TAU[l, k] + Theta_m[l * num_ps_per_TTD + p, k]
            TAU_list.append(TAU)

        if FFT_based:
            # x 1/eta instead of max_delay for FFT computation
            delay_list = np.arange(grids)/grids/eta
            stop_index = int(np.ceil(grids*max_delay*eta))
            # print(grids)
            # print(stop_index) # should equals grid before expansion, e.g. 256
            f_bias = fc - (num_sc-1)/2*eta # frequency of the initial subcarrier
            factor_list = np.exp(-1j*2*np.pi*f_bias*delay_list)
            TAU_list = np.array(TAU_list)
            sequences = np.conjugate(TAU_list)*np.linalg.norm(FBBs,axis=(1,2),keepdims=True)**2
            for l in range(num_TTD):
                for k in range(num_rf):
                    sequence = sequences[:,l,k]
                    fft_responses = np.fft.fft(sequence,grids)*factor_list
                    # only keep the beginning segment of fft responses due to the max_delay constraint
                    if partial_fft:
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
                            f = fc + (n-(num_sc-1)/2)*eta
                            fft_response = fft_response + np.conjugate(TAU_list[n][l, k]) * np.linalg.norm(
                                FBBs[n]) ** 2 * np.exp(-1j * 2 * np.pi * f * delay)
                        fft_responses.append(fft_response)
                    T[l, k] = delay_list[np.argmax(np.real(fft_responses))]
            # plt.figure()
            # plt.plot(np.real(fft_responses))
        # print(fft_responses)
            
        end = time.time()
        time_elapse = time_elapse + end - start
                
        HBFs = np.zeros((num_sc, num_antenna_bs, num_stream)) + 1j * np.zeros((num_sc, num_antenna_bs, num_stream))
        for n in range(num_sc):
            f = fc + (n-(num_sc-1)/2)*eta
            HBF = (A * np.kron(np.exp(-1j * 2 * np.pi * f * T), np.ones((num_ps_per_TTD, 1)))).dot(FBBs[n])
            # normalization
            HBF = HBF / np.linalg.norm(HBF) * np.sqrt(num_stream)
            HBFs[n] = HBF
        object_list.append(np.linalg.norm(HBFs - FDBs) ** 2 / np.product(np.shape(HBFs)))
        HBFs_list.append(HBFs)
        A_list.append(A)
        T_list.append(T)
        
        if early_stop:
            if i>=1:
                delta_object = object_list[-2]-object_list[-1]
                if (delta_object>0) & (delta_object<epsilon):
                    break

    return np.array(A_list), np.array(T_list), np.array(HBFs_list), np.array(object_list), time_elapse, i



def ZF(H,num_antenna_bs,num_antenna_ue,num_stream,num_user,p):
    assert H.shape == (num_antenna_ue,num_antenna_bs,num_user)
    H = np.reshape(H,(num_antenna_bs,num_user))
    H = np.transpose(H)
    V = np.linalg.pinv(H)
    # H.dot(V) = I
    V = np.expand_dims(V, axis=1)
    
    trace_VV = 0
    for user in range(num_user):
        trace_VV = trace_VV + np.trace(V[:, :, user].dot(np.transpose(np.conjugate(V[:, :, user]))))
    energy_scale = np.sqrt(p/np.real(trace_VV))
    V = energy_scale*V 
    
    return V # np.linalg.norm(V)=p

def update_U(H,V,num_user,num_antenna_ue,sigma_2,p):
    U = list()
    trace_VV = 0
    for user in range(num_user):
        trace_VV = trace_VV + np.trace(V[:, :, user].dot(np.transpose(np.conjugate(V[:, :, user]))))
    for user in range(num_user):
        HVVH = np.zeros([num_antenna_ue,num_antenna_ue],dtype=np.complex64)
        for k in range(num_user):
            HV = H[:, :, user].dot(V[:, :, k])
            HVVH = HVVH + HV.dot(np.transpose(np.conjugate(HV)))
        inverse_temp = np.linalg.inv(sigma_2/p*trace_VV*np.eye(num_antenna_ue) + HVVH)
        U_this_user = (inverse_temp.dot(H[:, :, user])).dot(V[:, :, user])
        U.append(U_this_user)
    U = np.stack(U, axis=-1)  
    return U

def update_W(H,U,V,num_user,num_stream):
    W = list()
    for user in range(num_user):
        HV = H[:, :, user].dot(V[:, :, user])
        W_this_user = np.linalg.inv(np.eye(num_stream, dtype=np.complex64) - np.transpose(np.conjugate(U[:, :, user])).dot(HV))
        W.append(W_this_user)
    W = np.stack(W, -1) 
    return W

def update_V(H,U,W,num_antenna_bs,num_user,sigma_2,p):
    temp_B = np.zeros([num_antenna_bs, num_antenna_bs], dtype=np.complex64)
    for user in range(num_user):
        HHU = np.transpose(np.conjugate(H[:, :, user])).dot(U[:, :, user])
        trace_UWU = sigma_2/p*np.trace((U[:, :, user].dot(W[:, :, user])).dot(np.transpose(np.conjugate(U[:, :, user]))))
        temp_B = temp_B + trace_UWU*np.eye(num_antenna_bs,dtype=np.complex64) + (HHU.dot(W[:, :, user])).dot(np.transpose(np.conjugate(HHU)))
    temp_B_inverse = np.linalg.inv(temp_B)
    V = list()
    for user in range(num_user):
        HHUW = (np.transpose(np.conjugate(H[:, :, user])).dot(U[:, :, user])).dot(W[:, :, user])
        V_this_user = temp_B_inverse.dot(HHUW)
        V.append(V_this_user)
    V = np.stack(V, axis=-1)

    trace_VV = 0
    for user in range(num_user):
        trace_VV = trace_VV + np.trace(V[:, :, user].dot(np.transpose(np.conjugate(V[:, :, user]))))
    energy_scale = np.sqrt(p/np.real(trace_VV))
    V = energy_scale*V # np.linalg.norm(V)=p
    
    return V
