
from bisect import bisect_left

import numpy as np

"""
DVB-S2
modulation type and code rate selection:
QPSK 1/4 1/3 2/5 1/2 3/5 2/3 3/4 4/5 5/6 8/9 9/10
8PSK 3/5 2/3 3/4 4/5 5/6 8/9 9/10
16APSK 2/3 3/4 4/5 5/6 8/9 9/10
32APSK 3/4 4/5 5/6 8/9 9/10
"""

"""
k=17.1:modulation type and code rate selection:
QPSK 1/4 1/3 2/5 1/2 3/5 2/3 3/4 4/5 5/6 
8PSK 3/5 2/3 3/4
16APSK 2/3
"""

MAXSNR = 100

PACKET_LENGTH = 12000
END_BER = 1e-8

SNR_DIVIDE_60 = ([0, 0.5, 1, 1.5, 2],
                 [1.25, 1.5, 1.75, 2, 2.25],
                 [2.67, 2.89, 3.11, 3.56],
                 [4.5, 4.72, 5.17, 5.61, 5.83],
                 [9.44, 9.89, 10.33, 11.67],
                 [11, 11.33, 12],
                 )

BIT_ERROR_60 = ([7.66e-4, 1.16e-4, 9.81e-5, 1.16e-5, 2.47e-6, 1e-10],
                [2.99e-3, 1.2e-3, 2.56e-4, 1.09e-4, 8.29e-5, 1e-10],
                [3.19e-3, 8.68e-4, 2.28e-4, 2.44e-5, 1e-10],
                [9.13e-3, 3.19e-3, 7.07e-4, 2.90e-4, 6.04e-5, 1e-10],
                [1.47e-3, 7.32e-4, 2.38e-4, 1.18e-6, 1e-10],
                [3.63e-4, 1.78e-5, 3.15e-6, 1e-10]
                )

SNR_DIVIDE_70 = ([-1, -0.67, -0.33],
                 [0.44, 0.67, 0.89, 1.11, 1.33],
                 [1.67, 1.72, 1.83, 2],
                 [3.61, 3.72, 3.83, 3.94, 4.06, 4.17],
                 [6.44, 6.89, 7.33, 7.78],
                 [7, 7.33, 7.56, 7.78],
                 [8.78, 9.22, 9.67, 9.89, 10.11, 10.56],
                 [11, 11.44, 11.89, 12.11])

BIT_ERROR_70 = ([1.92e-3, 3.09e-4, 6.52e-5, 1e-10],
                [1.31e-3, 3.87e-4, 1.72e-4, 3.31e-5, 1.20e-6, 1e-10],
                [2.09e-3, 7.83e-4, 1.70e-4, 4.40e-5, 1e-10],
                [7.31e-4, 4.82e-4, 2.63e-4, 2.13e-4, 8.63e-5, 1.30e-6, 1e-10],
                [1.37e-3, 2.31e-4, 5.41e-5, 8.23e-7, 1e-10],
                [6.61e-4, 1.79e-4, 6.61e-5, 4.86e-6, 1e-10],
                [1.61e-3, 6.35e-4, 1.49e-4, 9.89e-5, 1.40e-5, 1.93e-6, 1e-10],
                [1.35e-3, 6.82e-4, 2.61e-4, 8.60e-5, 1e-10])

SNR_DIVIDE_80 = ([-2.06, -1.56, -1.33, -1.11],
                 [-0.33, -0.11, 0.11],
                 [0.67, 0.89, 0.94, 1.06],
                 [2.33, 2.44, 2.56, 2.61, 2.67],
                 [4.17, 4.47, 4.53],
                 [4.83, 5, 5.17, 5.33],
                 [6.11, 6.33, 6.56, 6.78],
                 [7, 7.22, 7.44, 7.67],
                 [7.67, 7.89, 8.11, 8.33, 8.56, 8.78],
                 [10, 10.22, 10.44],
                 [13, 13.33, 14.78])

BIT_ERROR_80 = ([1.21e-3, 1.78e-4, 5.65e-5, 1.11e-6, 1e-10],
                [7.00e-4, 1.67e-4, 1.91e-5, 1e-10],
                [2.16e-3, 1.1e-4, 4.46e-5, 1.54e-7, 1e-10],
                [1.85e-3, 3.52e-4, 2.21e-4, 1.49e-4, 6.17e-7, 1e-10],
                [7.02e-4, 1.9e-4, 1.03e-7, 1e-10],
                [1.18e-3, 5.86e-4, 9.72e-5, 9.72e-7, 1e-10],
                [1.96e-3, 9.17e-4, 3.08e-5, 2.26e-6, 1e-10],
                [1.49e-3, 6.46e-4, 2.64e-4, 3.09e-6, 1e-10],
                [2.63e-3, 1.50e-3, 3.67e-4, 2.20e-4, 8.07e-5, 1.69e-5, 1e-10],
                [7.47e-4, 4.75e-4, 3.86e-5, 1e-10],
                [8.63e-4, 2.02e-4, 4.78e-7, 1e-10])

SNR_DIVIDE_90 = ([-2.5, -2.25, -2, -1.75],
                 [-1.25, -1, -0.75, -0.5],
                 [0, 0.12, 0.25, 0.38],
                 [1.5, 1.75, 2],
                 [3.06, 3.17, 3.22, 3.33, 3.44],
                 [4, 4.22, 4.67],
                 [4.83, 4.94, 5.06, 5.17, 5.28, 5.39],
                 [6, 6.11, 6.22],
                 [6.11, 6.33, 6.56, 6.78],
                 [7, 7.5, 8, 8.5],
                 [9, 9.44, 9.88],
                 [11, 11.5, 12, 12.5],
                 [15, 15.75, 16.25, 16.5])

BIT_ERROR_90 = ([9.05e-3, 1.50e-3, 1.53e-4, 2.47e-6, 1e-10],
                [2.73e-2, 4.24e-3, 3.34e-4, 4.26e-6, 1e-10],
                [7.93e-3, 1.70e-3, 4.77e-4, 4.48e-6, 1e-10],
                [7.06e-3, 3.70e-4, 6.30e-6, 1e-10],
                [1.65e-3, 7.75e-4, 2.91e-4,1.34e-4, 7.56e-7, 1e-10],
                [1.05e-3, 4.31e-5, 9.26e-8, 1e-10],
                [8.1e-3, 4.32e-3, 2.26e-3, 5.68e-4, 3.33e-4, 2.07e-5, 1e-10],
                [4.56e-4, 3.26e-4, 4.24e-7, 1e-10],
                [9.58e-3, 3.3e-3, 6.77e-4, 2e-6, 1e-10],
                [2.32e-3, 3.5e-4, 4.27e-6, 1.03e-7, 1e-10],
                [1.15e-3, 5.58e-5, 1.39e-7, 1e-10],
                [3.46e-3, 2.51e-3, 5.91e-4, 6.23e-5, 1e-10],
                [2.87e-4, 3.06e-4, 8.25e-5, 1.39e-7, 1e-10])


def get_snr_ber_divide(snr, mc, angle):

    if angle < 60:
        if mc >= len(SNR_DIVIDE_60):
            mc = len(SNR_DIVIDE_60) - 1

        snr_divide = SNR_DIVIDE_60[mc]
        ber_divide = BIT_ERROR_60[mc]

    elif 60 <= angle < 70:
        if mc >= len(SNR_DIVIDE_70):
            mc = len(SNR_DIVIDE_70) - 1

        snr_divide = SNR_DIVIDE_70[mc]
        ber_divide = BIT_ERROR_70[mc]

    elif 70 <= angle < 80:
        if mc >= len(SNR_DIVIDE_80):
            mc = len(SNR_DIVIDE_80) - 1

        snr_divide = SNR_DIVIDE_80[mc]
        ber_divide = BIT_ERROR_80[mc]

    else:
        if mc >= len(SNR_DIVIDE_90):
            mc = len(SNR_DIVIDE_90) - 1

        snr_divide = SNR_DIVIDE_90[mc]
        ber_divide = BIT_ERROR_90[mc]

    if snr <= snr_divide[0]:
        return np.log10(ber_divide[0])

    if snr >= snr_divide[-1]:
        return np.log10(ber_divide[-2])

    idx = bisect_left(snr_divide, snr)

    if snr == snr_divide[idx]:
        return np.log10(ber_divide[idx])

    prev_idx = idx - 1
    next_idx = idx

    log_ber_prev = np.log10(ber_divide[prev_idx])
    log_ber_next = np.log10(ber_divide[next_idx])

    snr_prev = snr_divide[prev_idx]
    snr_next = snr_divide[next_idx]

    log_ber = log_ber_prev + (log_ber_next - log_ber_prev) * (snr - snr_prev) / (snr_next - snr_prev)

    return log_ber


def ber_cal(snr, mc, angle):
   if angle < 60:
       if mc >= len(SNR_DIVIDE_60):
           mc = len(SNR_DIVIDE_60) - 1
       tmp = bisect_left(SNR_DIVIDE_60[mc], snr) - 1  # 前一个编号
       if tmp == len(SNR_DIVIDE_60[mc]) - 1:  # 队列最后一个
           ber = BIT_ERROR_60[mc][tmp]
       else:
           ber = pow(10, (np.log10(BIT_ERROR_60[mc][tmp]) + (
                   np.log10(BIT_ERROR_60[mc][tmp + 1]) - np.log10(BIT_ERROR_60[mc][tmp])) * (
                                  snr - SNR_DIVIDE_60[mc][tmp]) / (
                                  SNR_DIVIDE_60[mc][tmp + 1] - SNR_DIVIDE_60[mc][tmp])))
   elif 60 <= angle < 70:
       if mc >= len(SNR_DIVIDE_70):
           mc = len(SNR_DIVIDE_70) - 1
       tmp = bisect_left(SNR_DIVIDE_70[mc], snr) - 1
       if tmp == len(SNR_DIVIDE_70[mc]) - 1:
           ber = BIT_ERROR_70[mc][tmp]
       else:
           ber = pow(10, (np.log10(BIT_ERROR_70[mc][tmp]) + (
                   np.log10(BIT_ERROR_70[mc][tmp + 1]) - np.log10(BIT_ERROR_70[mc][tmp])) * (
                                  snr - SNR_DIVIDE_70[mc][tmp]) / (
                                  SNR_DIVIDE_70[mc][tmp + 1] - SNR_DIVIDE_70[mc][tmp])))
   elif 70 <= angle < 80:
       if mc >= len(SNR_DIVIDE_80):
           mc = len(SNR_DIVIDE_80) - 1
       tmp = bisect_left(SNR_DIVIDE_80[mc], snr) - 1
       if tmp == len(SNR_DIVIDE_80[mc]) - 1:
           ber = BIT_ERROR_80[mc][tmp]
       else:
           ber = pow(10, (np.log10(BIT_ERROR_80[mc][tmp]) + (
                   np.log10(BIT_ERROR_80[mc][tmp + 1]) - np.log10(BIT_ERROR_80[mc][tmp])) * (
                                  snr - SNR_DIVIDE_80[mc][tmp]) / (
                                  SNR_DIVIDE_80[mc][tmp + 1] - SNR_DIVIDE_80[mc][tmp])))
   else:
       if mc >= len(SNR_DIVIDE_90):
           mc = len(SNR_DIVIDE_90) - 1
       tmp = bisect_left(SNR_DIVIDE_90[mc], snr) - 1
       if tmp == len(SNR_DIVIDE_90[mc]) - 1:
           ber = BIT_ERROR_90[mc][tmp]
       else:
           ber = pow(10, (np.log10(BIT_ERROR_90[mc][tmp]) + (
                   np.log10(BIT_ERROR_90[mc][tmp + 1]) - np.log10(BIT_ERROR_90[mc][tmp])) * (
                                  snr - SNR_DIVIDE_90[mc][tmp]) / (
                                  SNR_DIVIDE_90[mc][tmp + 1] - SNR_DIVIDE_90[mc][tmp])))

   return np.log10(ber)




def env_reward_cal(snr, mc, angle):
    """
    mc:0-12
        QPSK 1/4 1/3 2/5 1/2 3/5 2/3 3/4 4/5 5/6
        8PSK 3/5 2/3 3/4
        16APSK 2/3
    angle:卫星仰角
    """
    if angle < 60:
        if snr <= 1.0:
            return 0
        else:
            snr_divide = [1.0, 2.1, 3.3, 5.8, 10.6, 11.2, MAXSNR]
            effi = [2/4, 2/3, 4/5, 2/2, 6/5, 1.33]
            n = bisect_left(snr_divide, snr) - 1
            if mc <= n:
                tmp = bisect_left(SNR_DIVIDE_60[mc], snr) - 1
                if tmp == len(SNR_DIVIDE_60[mc]) - 1:
                    ber = BIT_ERROR_60[mc][tmp]
                else:
                    ber = pow(10, (np.log10(BIT_ERROR_60[mc][tmp])+(np.log10(BIT_ERROR_60[mc][tmp+1])-np.log10(BIT_ERROR_60[mc][tmp])) * (snr-SNR_DIVIDE_60[mc][tmp]) / (SNR_DIVIDE_60[mc][tmp+1]-SNR_DIVIDE_60[mc][tmp])))
                per = pow((1-ber), PACKET_LENGTH)
                return effi[mc] * per
            else:
                return 0
    elif 60 <= angle < 70:
        if snr <= -0.4:
            return 0
        else:
            snr_divide = [-0.4, 1.0, 1.9, 4.1, 7.2, 7.5, 9.9, 12.1, MAXSNR]
            effi = [2/4, 2/3, 4/5, 2/2, 6/5, 4/3, 6/4, 1.6]
            n = bisect_left(snr_divide, snr) - 1
            if mc <= n:
                tmp = bisect_left(SNR_DIVIDE_70[mc], snr) - 1
                if tmp == len(SNR_DIVIDE_70[mc]) - 1:
                    ber = BIT_ERROR_70[mc][tmp]
                else:
                    ber = pow(10, (np.log10(BIT_ERROR_70[mc][tmp]) + (
                            np.log10(BIT_ERROR_70[mc][tmp + 1]) - np.log10(BIT_ERROR_70[mc][tmp])) * (
                                       snr - SNR_DIVIDE_70[mc][tmp]) / (
                                       SNR_DIVIDE_70[mc][tmp + 1] - SNR_DIVIDE_70[mc][tmp])))
                per = pow((1 - ber), PACKET_LENGTH)
                return effi[mc] * per
            else:
                return 0
    elif 70 <= angle < 80:
        if snr <= -1.4:
            return 0
        else:
            snr_divide = [-1.4, 0.0, 0.9, 2.7, 4.5, 5.2, 6.5, 10.4, 13.5, MAXSNR]
            effi = [2/4, 2/3, 4/5, 2/2, 6/5, 4/3, 6/4, 8/5, 10/6, 9/5, 2]
            n = bisect_left(snr_divide, snr) - 1
            if mc <= n:
                tmp = bisect_left(SNR_DIVIDE_80[mc], snr) - 1
                if tmp == len(SNR_DIVIDE_80[mc]) - 1:
                    ber = BIT_ERROR_80[mc][tmp]
                else:
                    ber = pow(10, (np.log10(BIT_ERROR_80[mc][tmp]) + (
                            np.log10(BIT_ERROR_80[mc][tmp + 1]) - np.log10(BIT_ERROR_80[mc][tmp])) * (
                                       snr - SNR_DIVIDE_80[mc][tmp]) / (
                                       SNR_DIVIDE_80[mc][tmp + 1] - SNR_DIVIDE_80[mc][tmp])))
                per = pow((1 - ber), PACKET_LENGTH)
                return effi[mc] * per
            else:
                return 0
    else:
        if snr <= -1.9:
            return 0
        else:
            snr_divide = [-1.9, -0.6, 0.3, 1.9, 3.4, 4.2, 5.4, 6.2, 6.7, 7.7, 9.4, 12.4, 16.2, MAXSNR]
            effi = [2/4, 2/3, 4/5, 2/2, 6/5, 4/3, 6/4, 8/5, 10/6, 9/5, 2, 9/4, 8/3]
            n = bisect_left(snr_divide, snr) - 1
            if mc <= n:
                tmp = bisect_left(SNR_DIVIDE_90[mc], snr) - 1
                if tmp == len(SNR_DIVIDE_90[mc]) - 1:
                    ber = BIT_ERROR_90[mc][tmp]
                else:
                    ber = pow(10, (np.log10(BIT_ERROR_90[mc][tmp]) + (
                            np.log10(BIT_ERROR_90[mc][tmp + 1]) - np.log10(BIT_ERROR_90[mc][tmp])) * (
                                       snr - SNR_DIVIDE_90[mc][tmp]) / (
                                       SNR_DIVIDE_90[mc][tmp + 1] - SNR_DIVIDE_90[mc][tmp])))
                per = pow((1 - ber), PACKET_LENGTH)
                return effi[mc] * per
            else:
                return 0


def spec_effi_modulation(snr, angle):
    if angle < 60:
        if snr <= 1.0:
            return 0, 0
        elif snr > 11.2:
            return 1.33, 5
        else:
            snr_divide = [1.0, 2.1, 3.3, 5.8, 10.6, 11.2]
            effi = [2/4, 2/3, 4/5, 2/2, 6/5, 1.33]
            n = bisect_left(snr_divide, snr) - 1
            return effi[n], n
    elif 60 <= angle < 70:
        if snr <= -0.4:
            return 0, 0
        elif snr > 12.1:
            return 1.6, 7
        else:
            snr_divide = [-0.4, 1.0, 1.9, 4.1, 7.2, 7.5, 9.9, 12.1]
            effi = [2/4, 2/3, 4/5, 2/2, 6/5, 4/3, 6/4, 1.6]
            n = bisect_left(snr_divide, snr) - 1
            return effi[n], n
    elif 70 <= angle < 80:
        if snr <= -1.4:
            return 0, 0
        elif snr > 13.5:
            return 2, 10
        else:
            snr_divide = [-1.4, 0.0, 0.9, 2.7, 4.5, 5.2, 6.5, 10.4, 13.5]
            effi = [2/4, 2/3, 4/5, 2/2, 6/5, 4/3, 6/4, 8/5, 10/6, 9/5, 2]
            n = bisect_left(snr_divide, snr) - 1
            return effi[n], n
    else:
        if snr <= -1.9:
            return 0, 0
        elif snr > 16.2:
            return 2.67, 12
        else:
            snr_divide = [-1.9, -0.6, 0.3, 1.9, 3.4, 4.2, 5.4, 6.2, 6.7, 7.7, 9.4, 12.4, 16.2]
            effi = [2/4, 2/3, 4/5, 2/2, 6/5, 4/3, 6/4, 8/5, 10/6, 9/5, 2, 9/4, 8/3]
            n = bisect_left(snr_divide, snr) - 1
            return effi[n], n

