import torch
from satellite_param import *
from rician_data_collect import *
from config import Config

# water-filling


def water_filling(jammer_power, noise_power, total_power, g_i):
    K = len(jammer_power)
    channel_gain = []
    for i in range(K):
        channel_gain.append(g_i[i] / (noise_power + jammer_power[i]))
    channels_sortindex = np.argsort(channel_gain)[::-1]
    channels_sorted = np.array(channel_gain)[channels_sortindex]
    h = 1 / channels_sorted
    if (K - 1) * h[K - 1] - sum(h[np.arange(0, K - 2)]) < total_power:
        idx = K - 1
    else:
        idx_max = K - 1
        idx_min = 0

        while True:
            idx_mid = math.floor(0.5 * (idx_max + idx_min))
            w_tmp = idx_mid
            h_tmp = h[idx_mid]
            if w_tmp * h_tmp - sum(h[np.arange(0, idx_mid - 1)]) < total_power:
                idx_min = idx_mid
            else:
                idx_max = idx_mid
            if idx_min + 1 == idx_max:
                idx = idx_min
                break

    w_filled = idx + 1
    h_filled = (total_power + sum(h[np.arange(0, idx + 1)])) / w_filled
    p_allocate = np.zeros([K])
    p_allocate[np.arange(0, idx + 1)] = h_filled - h[np.arange(0, idx + 1)]

    powers_opt = np.zeros([K])
    powers_opt[channels_sortindex[np.arange(0, K)]] = p_allocate

    return powers_opt


actor_net = torch.load("./results/model/actor_model.pt")
multiplier_net = torch.load("./results/model/multiplier_model.pt")
critic_net = torch.load("./results/model/Qnet_model.pt")

# 用户终端及干扰者位置
lat = 38.913611
lon = -77.013222
loc = latlon2ecef(lat, lon)
lon_j = lon + 85 / (math.pi * math.cos(lat * math.pi / 180) * 6378) * 180
loc_j = latlon2ecef(lat, lon_j)

sender = np.array(loc)
jammer = np.array(loc_j)

e_us = []
e_us_plus = []
e_us_minus = []

g_js = []
g_js_plus = []
g_js_minus = []

g_sj = []
g_sj_plus = []
g_sj_minus = []

power_jammer_dbw = 30
power_user_dbw = -15
g_t = 9.25
g_su = g_t + 10 * math.log10(300)
g_us = 35

power_jammer = math.pow(10, power_jammer_dbw / 10)
power_user = math.pow(10, (g_us + g_su + power_user_dbw) / 10)

noise_power = 1.38064852e-17 * 300 * 1150 * 1150 * 50

v = np.linspace(0, 360, 5000)

c_opts_rl = []
powers_opts_pm = []
c_opts_pm = []
c_opts_pmt = []
c_opts_single = []
c_opts_singlet = []


j_net = []
c_net = []
c_true = []
m_net = []
powers_sum = []

v_divide = [22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0, 202.5, 225.0, 247.5, 270.0, 292.5, 315.0, 337.5]
v0 = 40
for i in range(len(v)):
    phi = v[i] * 108 / 1440
    n = bisect_left(v_divide, v[i])
    # 计算卫星集中卫星位置
    if n < 8:
        satellite_position = ecef_position(v[i] + v0 - 21.6 * n, phi, rs=7528, w=0, omega=247.5, i=53)
        satellite_pos_plus = ecef_position(v[i] + v0 - 21.6 * n + 7.2, phi, rs=7528, w=0, omega=247.5, i=53)
        satellite_pos_minus = ecef_position(v[i] + v0 - 21.6 * n - 7.2, phi, rs=7528, w=0, omega=247.5, i=53)

    else:
        satellite_position = ecef_position(v[i] + v0 - 21.6 * n - 10.6, phi, rs=7528, w=0, omega=270, i=53)
        satellite_pos_plus = ecef_position(v[i] + v0 - 21.6 * n - 10.6 + 7.2, phi, rs=7528, w=0, omega=270, i=53)
        satellite_pos_minus = ecef_position(v[i] + v0 - 21.6 * n - 10.6 - 7.2, phi, rs=7528, w=0, omega=270, i=53)

    e_us_plus = (math.pi / 2 - cal_angle(sender, satellite_pos_plus - sender)) * 180 / math.pi
    e_us_minus = (math.pi / 2 - cal_angle(sender, satellite_pos_minus - sender)) * 180 / math.pi
    e_us = (math.pi / 2 - cal_angle(sender, satellite_position - sender)) * 180 / math.pi

    # 判断卫星是否可见
    if e_us_plus >= 30:
        theta_plus = cal_angle(satellite_pos_plus - jammer, satellite_position - jammer)
        g_js_plus.append(antenna_gain_jammer(3, 0.3 / 14, theta_plus * 180 / math.pi))
    else:
        # 卫星不可见时不进行功率分配，将干扰者在其方向上的天线增益设为很大的值
        g_js_plus.append(100)
    if e_us_minus >= 30:
        theta_minus = cal_angle(satellite_pos_minus - jammer, satellite_position - jammer)
        g_js_minus.append(antenna_gain_jammer(3, 0.3 / 14, theta_minus * 180 / math.pi))
    else:
        g_js_minus.append(100)
    g_js.append(51.3)

    theta_upj = cal_angle(sender - satellite_pos_plus, jammer - satellite_pos_plus)
    g_sj_plus.append(antenna_gain_sat(0.7, 0.3 / 14, theta_upj * 180 / math.pi))
    theta_umj = cal_angle(sender - satellite_pos_minus, jammer - satellite_pos_minus)
    g_sj_minus.append(antenna_gain_sat(0.7, 0.3 / 14, theta_umj * 180 / math.pi))
    theta_usj = cal_angle(sender - satellite_position, jammer - satellite_position)
    g_sj.append(antenna_gain_sat(0.7, 0.3 / 14, theta_usj * 180 / math.pi))

    d_u = (sender - satellite_position).dot(sender - satellite_position)
    d_j = (jammer - satellite_position).dot(jammer - satellite_position)
    d_up = (sender - satellite_pos_plus).dot(sender - satellite_pos_plus)
    d_jp = (jammer - satellite_pos_plus).dot(jammer - satellite_pos_plus)
    d_um = (sender - satellite_pos_minus).dot(sender - satellite_pos_minus)
    d_jm = (jammer - satellite_pos_minus).dot(jammer - satellite_pos_minus)

    jammer_power_pm = [power_jammer * math.pow(10, (g_js[i]+g_sj[i]) / 10) * 1150 * 1150 / d_j,
                       power_jammer * math.pow(10, (g_js_plus[i]+g_sj_plus[i]) / 10) * 1150 * 1150 / d_jp,
                       power_jammer * math.pow(10, (g_js_minus[i]+g_sj_minus[i]) / 10) * 1150 * 1150 / d_jm]
    g_i = [1150 * 1150 / d_u, 1150 * 1150 / d_up, 1150 * 1150 / d_um]
    powers_opts_pm.append(water_filling(jammer_power_pm, noise_power, power_user, g_i))
    snr_wf = [
        10 * np.log10(powers_opts_pm[i][0] * g_i[0] / (jammer_power_pm[0] + noise_power)),
        10 * np.log10(powers_opts_pm[i][1] * g_i[1] / (jammer_power_pm[1] + noise_power)),
        10 * np.log10(powers_opts_pm[i][2] * g_i[2] / (jammer_power_pm[2] + noise_power))
    ]

    effi_ideal_sat1 = []
    effi_ideal_sat2 = []
    effi_fixed_sat1 = []
    effi_fixed_sat2 = []
    for mc in range(13):
        effi_ideal_sat1.append(env_reward_cal(snr_wf[1], mc, e_us_plus))
        effi_ideal_sat2.append(env_reward_cal(snr_wf[2], mc, e_us_minus))
        effi_fixed_sat1.append(env_reward_cal(snr_wf[1], mc, 30))
        effi_fixed_sat2.append(env_reward_cal(snr_wf[2], mc, 30))
    c_opts_pm.append([np.max(effi_ideal_sat1), np.max(effi_ideal_sat2)])
    c_opts_pmt.append([np.max(effi_fixed_sat1), np.max(effi_fixed_sat2)])

    state = [
        power_user * g_i[1] / (jammer_power_pm[1] + noise_power),
        power_user * g_i[2] / (jammer_power_pm[2] + noise_power),
    ]
    for j in range(len(state)):

        if state[j] <= 0.001:
            state[j] = -3 / Config.SNR_NORM_FACTOR
        else:
            state[j] = np.log10(state[j]) / Config.SNR_NORM_FACTOR
    state.append(e_us_plus / Config.ANGLE_NORM_FACTOR)
    state.append(e_us_minus / Config.ANGLE_NORM_FACTOR)

    state_rl = torch.tensor(state, dtype=torch.float).unsqueeze(0)
    action_rl = actor_net(state_rl)
    action_np = action_rl.detach().squeeze().numpy()
    snr = state[0:2]

    snr_db_rl = [10 * snr[0] * Config.SNR_NORM_FACTOR + 10 * np.log10(action_np[0]),
                 10 * snr[1] * Config.SNR_NORM_FACTOR + 10 * np.log10(action_np[1])]
    mc_choose = []
    for k in range(2):
        action_np[k] = min(1, action_np[k])
        mc_choose.append(critic_net.select_action(state_rl.squeeze()[k].unsqueeze(0).unsqueeze(0),
                                                  state_rl.squeeze()[k+2].unsqueeze(0).unsqueeze(0),
                                                  torch.tensor(action_np[k]).unsqueeze(0).unsqueeze(0)))
    c_opts_rl.append(
        [
            env_reward_cal(snr_db_rl[0], mc_choose[0], e_us_plus),
            env_reward_cal(snr_db_rl[1], mc_choose[1], e_us_minus),
        ]
    )

    # single-sat
    angle_u = np.array([e_us_plus, e_us_minus])
    distance_j = [d_jp, d_jm]
    g_j = [g_js_plus[i]+g_sj_plus[i], g_js_minus[i]+g_sj_minus[i]]
    index = np.argmax(angle_u)
    jammer_power = power_jammer * math.pow(10, g_j[index] / 10) * 1150 * 1150 / distance_j[index]
    g_u = 1150 * 1150 / angle_u[index]
    snr_single = 10 * np.log10(power_user * g_u / (jammer_power + noise_power))

    effi_ideal = []
    effi_fixed = []
    for mc in range(13):
        effi_ideal.append(env_reward_cal(snr_single, mc, angle_u[index]))
        effi_fixed.append(env_reward_cal(snr_single, mc, 30))

    c_opts_single.append(np.max(effi_ideal))
    c_opts_singlet.append(np.max(effi_fixed))


print("data-rate-ave-rl:", np.mean(np.array(c_opts_rl)) * 2)
print("data-rate-ave-wf-opt:", np.mean(np.array(c_opts_pm)) * 2)
print("data-rate-ave-wf:", np.mean(np.array(c_opts_pmt)) * 2)
print("single_sat_opt:", np.mean(c_opts_single))
print("single_sat:", np.mean(c_opts_singlet))











