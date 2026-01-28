import math
import numpy as np

# 计算卫星的ECEF坐标
def ecef_position(v, phi=0, rs=7528, w=0, omega=247.5, i=53):
    phi = phi * math.pi / 180
    w = w * math.pi / 180
    v = v * math.pi / 180
    omega = omega * math.pi / 180
    i = i * math.pi / 180
    position = [0, 0, 0]
    position[0] = rs * (math.cos(phi) * (math.cos(w + v) * math.cos(omega) - math.sin(w + v) * math.cos(i) * math.sin(omega))
                        + math.sin(phi) * (math.cos(w + v) * math.sin(omega) + math.sin(w + v) * math.cos(i) * math.cos(omega)))
    position[1] = rs * (-math.sin(phi) * (math.cos(w + v) * math.cos(omega) - math.sin(w + v) * math.cos(i) * math.sin(omega))
                        + math.cos(phi) * (math.cos(w + v) * math.sin(omega) + math.sin(w + v) * math.cos(i) * math.cos(omega)))
    position[2] = rs * (math.sin(w + v) * math.sin(i))

    return np.array(position)

# 计算向量a与向量b间夹角
def cal_angle(a, b):
    la = np.sqrt(a.dot(a))
    lb = np.sqrt(b.dot(b))
    cos_angle = (a.dot(b)) / (la * lb)
    angle = np.arccos(cos_angle)

    return angle

# 将经纬度转换为ECEF坐标
def latlon2ecef(lat, lon):
    lat = lat / 180 * math.pi  # 纬度
    lon = lon / 180 * math.pi  # 经度
    position = [6378 * math.cos(lat) * math.cos(lon),
                6378 * math.cos(lat) * math.sin(lon),
                6378 * math.sin(lat)]

    return position

# 计算不同离轴角下的天线增益
def antenna_gain_jammer(D, lamda, theta):
    g_max = 20 * math.log10(D / lamda) + 8.4
    g1 = - 1 + 15 * math.log10(D / lamda)
    theta_m = 20 * lamda / D * math.sqrt(g_max - g1)
    theta_r = 15.85 * math.pow((D / lamda), -0.6)
    if 0 <= theta < theta_m:
        g = g_max - 0.0025 * math.pow((D / lamda * theta), 2)
    elif theta_m <= theta < theta_r:
        g = g1
    elif theta_r <= theta < 10:
        g = 29 - 25 * math.log10(theta)
    elif 10 <= theta < 34.1:
        g = 34 - 30 * math.log10(theta)
    elif 34.1 <= theta < 80:
        g = -12
    elif 80 <= theta < 120:
        g = -7
    else:
        g = -12

    return g

def antenna_gain_sat(D, lamda, theta):
    phi_b = math.sqrt(1200) / (D / lamda)
    g_max = 34.021212547196626
    # g_max = 20 * math.log10(D / lamda) + 8.4
    y = 1.5 * phi_b
    z = y * math.pow(10, 0.04 * (g_max - 6.75 - 5))
    if 0 <= theta < phi_b:
        g = g_max
    elif phi_b <= theta < 1.5 * phi_b:
        g = g_max - 3 * math.pow(theta / phi_b, 2)
    elif y <= theta < z:
        g = g_max - 6.75 - 25 * math.log10(theta / y)
    else:
        g = 5

    return g
