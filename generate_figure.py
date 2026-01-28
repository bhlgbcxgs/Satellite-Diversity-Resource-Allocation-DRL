import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


plt.rc('font', family='Times New Roman')

srN = [5, 10, 15]

rl = [2.1162188691358073, 2.9196087387142207, 3.682364645058377]
wf_l = [1.8075387971063375, 2.587323550848668, 3.197631481977924]
wf_opt = [2.0466855798723396, 2.8244865630483806, 3.612530166637987]
width = 1.3
srN_left = [srN[i]-width for i in range(3)]
srN_right = [srN[i]+width for i in range(3)]

fig, ax = plt.subplots(figsize=(6.5, 5))

bar1 = ax.bar(srN_right, rl, width=width, color='#5861ac', edgecolor='white', linewidth=1, label='Proposed DRL-based approach', alpha=0.9, hatch='/')
bar2 = ax.bar(srN_left, wf_l, width=width, color='#ffc080', edgecolor='white', linewidth=1, label='Waterfilling with fixed MCS', alpha=0.9,hatch='\\')
bar3 = ax.bar(srN, wf_opt, width=width, color='#f28080', edgecolor='white',linewidth=1, label='Waterfilling with ideal MCS', alpha=0.9)

plt.xlabel('Number of Satellites', fontsize=12)
plt.ylabel('Average effective SE (bits/s/Hz)', fontsize=12)
plt.axhline(1.3214475071124892, linewidth=1.5, linestyle='dashed', color='black', label='Single satellite with fixed MCS')
plt.axhline(1.5617964600305856, linewidth=1.5, linestyle='dashdot', color='black', label='Single satellite with ideal MCS')
categories = ['3-satellite', '4-satellite', '5-satellite']

plt.ylim(1, 4)
ax.set_xticks(srN)
ax.set_xticklabels(categories, fontsize=12)

plt.grid(True, linestyle='-', alpha=1)

plt.legend(handlelength=3.5, fontsize=12)
plt.tight_layout()
plt.show()