import random
from satellite_param import *
from bisect import bisect_left
from rician_data_collect import env_reward_cal
from rician_data_collect import get_snr_ber_divide

class EnvWF():

    def __init__(self, snr_norm, angle_norm):
        self.satellite_num = 2
        self.snr_norm = snr_norm
        self.angle_norm = angle_norm


    def step(self, state, action, mc):
        """
        :param action: power allocation of each channel[0~1] (2 * satellite_num--means and variances)
        :param state: channel gain of each channel(total power * g_i / (jammer_power + noise_power))---alloc total=1
        :return: reward: total spectral efficiency
        """

        spec_effi = []
        ber = []
        for i in range(self.satellite_num):
            snr = 10 * state[i] * self.snr_norm + 10 * np.log10(action[i])
            angle = self.angle_norm * state[i+self.satellite_num]
            spec_effi.append(env_reward_cal(snr, mc[i], angle))
            ber.append(get_snr_ber_divide(snr, mc[i], angle))

        return spec_effi, ber


    def ge_rnd_state(self, v):
        # 用户终端及干扰者位置
        lat = 38.913611
        lon = -77.013222
        loc = latlon2ecef(lat, lon)
        lon_j = lon + 85 / (math.pi * math.cos(lat * math.pi / 180) * 6378) * 180
        loc_j = latlon2ecef(lat, lon_j)

        sender = np.array(loc)
        jammer = np.array(loc_j)

        power_jammer_dbw = 30
        power_user_dbw = -15
        g_t = 9.25
        g_su = g_t + 10 * math.log10(300)
        g_us = 35

        power_jammer = math.pow(10, power_jammer_dbw / 10)
        power_user = math.pow(10, (g_us + g_su + power_user_dbw) / 10)
        noise_power = 1.38064852e-17 * 300 * 1150 * 1150 * 50

        v_divide = [22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0,
                    337.5]
        v0 = 40

        v = random.random() * 360
        phi = v * 108 / 1440
        n = bisect_left(v_divide, v)
        if n < 8:
            satellite_position = ecef_position(v + v0 - 21.6 * n, phi, rs=7528, w=0, omega=247.5, i=53)
            satellite_pos_plus = ecef_position(v + v0 - 21.6 * n + 7.2, phi, rs=7528, w=0, omega=247.5, i=53)
            satellite_pos_minus = ecef_position(v + v0 - 21.6 * n - 7.2, phi, rs=7528, w=0, omega=247.5, i=53)
            satellite_pos_west = ecef_position(v + v0 - 21.6 * n + 7.2, phi, rs=7528, w=0, omega=236.35, i=53)
            satellite_pos_east = ecef_position(v + v0 - 21.6 * n - 7.2, phi, rs=7528, w=0, omega=258.75, i=53)
        else:
            satellite_position = ecef_position(v + v0 - 21.6 * n - 10.6, phi, rs=7528, w=0, omega=270, i=53)
            satellite_pos_plus = ecef_position(v + v0 - 21.6 * n - 10.6 + 7.2, phi, rs=7528, w=0, omega=270, i=53)
            satellite_pos_minus = ecef_position(v + v0 - 21.6 * n - 10.6 - 7.2, phi, rs=7528, w=0, omega=270, i=53)
            satellite_pos_west = ecef_position(v + v0 - 21.6 * n - 10.6 + 7.2, phi, rs=7528, w=0, omega=258.75,
                                               i=53)
            satellite_pos_east = ecef_position(v + v0 - 21.6 * n - 10.6 - 7.2, phi, rs=7528, w=0, omega=281.35,
                                               i=53)

        e_us_plus = (math.pi / 2 - cal_angle(sender, satellite_pos_plus - sender)) * 180 / math.pi
        e_us_minus = (math.pi / 2 - cal_angle(sender, satellite_pos_minus - sender)) * 180 / math.pi
        e_us_west = (math.pi / 2 - cal_angle(sender, satellite_pos_west - sender)) * 180 / math.pi
        e_us_east = (math.pi / 2 - cal_angle(sender, satellite_pos_east - sender)) * 180 / math.pi

        # 判断卫星是否可见
        if e_us_plus >= 30:
            theta_plus = cal_angle(satellite_pos_plus - jammer, satellite_position - jammer)
            g_js_plus = antenna_gain_jammer(3, 0.3 / 14, theta_plus * 180 / math.pi)
        else:
            # 卫星不可见时不进行功率分配，将干扰者在其方向上的天线增益设为很大的值
            g_js_plus = 100
        if e_us_minus >= 30:
            theta_minus = cal_angle(satellite_pos_minus - jammer, satellite_position - jammer)
            g_js_minus = antenna_gain_jammer(3, 0.3 / 14, theta_minus * 180 / math.pi)
        else:
            g_js_minus = 100
        if e_us_west >= 30:
            theta_west = cal_angle(satellite_pos_west - jammer, satellite_position - jammer)
            g_js_west = antenna_gain_jammer(3, 0.3 / 14, theta_west * 180 / math.pi)
        else:
            g_js_west = 100
        if e_us_east >= 30:
            theta_east = cal_angle(satellite_pos_east - jammer, satellite_position - jammer)
            g_js_east = antenna_gain_jammer(3, 0.3 / 14, theta_east * 180 / math.pi)
        else:
            g_js_east = 100
        # 主卫星增益为51.3dB
        g_js = 51.3

        theta_upj = cal_angle(sender - satellite_pos_plus, jammer - satellite_pos_plus)
        g_sj_plus = antenna_gain_sat(0.7, 0.3 / 14, theta_upj * 180 / math.pi)
        theta_umj = cal_angle(sender - satellite_pos_minus, jammer - satellite_pos_minus)
        g_sj_minus = antenna_gain_sat(0.7, 0.3 / 14, theta_umj * 180 / math.pi)
        theta_uwj = cal_angle(sender - satellite_pos_west, jammer - satellite_pos_west)
        g_sj_west = antenna_gain_sat(0.7, 0.3 / 14, theta_uwj * 180 / math.pi)
        theta_uej = cal_angle(sender - satellite_pos_east, jammer - satellite_pos_east)
        g_sj_east = antenna_gain_sat(0.7, 0.3 / 14, theta_uej * 180 / math.pi)
        theta_usj = cal_angle(sender - satellite_position, jammer - satellite_position)
        g_sj = antenna_gain_sat(0.7, 0.3 / 14, theta_usj * 180 / math.pi)
        # 用户及干扰者到各卫星距离（平方）
        d_j = (jammer - satellite_position).dot(jammer - satellite_position)
        d_up = (sender - satellite_pos_plus).dot(sender - satellite_pos_plus)
        d_jp = (jammer - satellite_pos_plus).dot(jammer - satellite_pos_plus)
        d_um = (sender - satellite_pos_minus).dot(sender - satellite_pos_minus)
        d_jm = (jammer - satellite_pos_minus).dot(jammer - satellite_pos_minus)

        # 干扰者发射到各卫星的干扰信号功率
        jammer_power = [power_jammer * math.pow(10, (g_js+g_sj) / 10) * 1150 * 1150 / d_j,
                        power_jammer * math.pow(10, (g_js_plus+g_sj_plus) / 10) * 1150 * 1150 / d_jp,
                        power_jammer * math.pow(10, (g_js_minus+g_sj_minus) / 10) * 1150 * 1150 / d_jm]

        snr = [power_user * 1150 * 1150 / d_up / (jammer_power[1] + noise_power),
               power_user * 1150 * 1150 / d_um / (jammer_power[2] + noise_power)]

        state_sorted = snr
        for i in range(len(state_sorted)):
            if state_sorted[i] < 0.001:
                state_sorted[i] = -3 / self.snr_norm
            else:
                state_sorted[i] = np.log10(state_sorted[i]) / self.snr_norm
        state_sorted.append(e_us_plus / self.angle_norm)
        state_sorted.append(e_us_minus / self.angle_norm)
        # print(state_sorted)
        return state_sorted


