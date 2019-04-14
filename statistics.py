# -*-coding: utf-8-*-
# Author: Yilin Zhang
'''
This script is for doing statistics, to give a reference for the decision of
data selection rule.
'''

import pickle
import re
import matplotlib.pyplot as plt


def interval_stats(raw_list, interval, number):
    stats_list = [0] * number
    for n in raw_list:
        categoried = False
        for i in range(number - 1):
            if n >= interval * i and n < interval * (i + 1):
                stats_list[i] += 1
                categoried = True
                break
        if not categoried:
            stats_list[-1] += 1
    return stats_list


# --------- Do statistics for all ntoes ----------

# with open('./pickle/itv_freq.pkl', 'rb') as f:
#     itv_freq = pickle.load(f)
# plt.bar(list(itv_freq.keys()), itv_freq.values(), color='g')

# with open('./pickle/dur_freq.pkl', 'rb') as f:
#     dur_freq = pickle.load(f)
# plt.bar(list(dur_freq.keys()), dur_freq.values(), color='g')

# with open('./pickle/res_freq.pkl', 'rb') as f:
#     res_freq = pickle.load(f)
# plt.bar(list(res_freq.keys()), res_freq.values(), color='g')

# --------- Do statistics for phrase information ----------
# itv_break = 0
# dur_break = 0
# res_break = 0
# with open("phrase_info.log") as f:
#     for line in f:
#         if re.search('^phrase break: interval', line):
#             itv_break += 1
#         elif re.search('^phrase break: duration', line):
#             dur_break += 1
#         elif re.search('^phrase break: rest', line):
#             res_break += 1
# print('itv_break:', itv_break)
# print('dur_break:', dur_break)
# print('res_break:', res_break)

# --------- Do statistics for all phrases ----------

with open('./pickle/phrase_lengths.pkl', 'rb') as f:
    phrase_lengths = pickle.load(f)

n, bins, patches = plt.hist(
    phrase_lengths, bins=100, range=(0, 100), facecolor='g', alpha=0.75)
delete = sum(n[0:3])
reserve = sum(n[3:])
print(delete)
print(reserve)
# --------- Do statistics for all special notes ----------

# # Read duration and interval information from log file.
# pos_intervals = []
# neg_intervals = []
# durations = []
# rests = []
# with open("info16.log") as f:
#     for line in f:
#         if re.search('^big_pos_itv:', line):
#             pos_intervals.append(int(line.split()[1]))
#         elif re.search('^big_neg_itv:', line):
#             neg_intervals.append(int(line.split()[1]))
#         elif re.search('^long_dur:', line):
#             durations.append(int(line.split()[1]))
#         elif re.search('^long_res:', line):
#             rests.append(int(line.split()[1]))

# # # Do statistics
# # pos_itv_stats_list = interval_stats(pos_intervals, 10, 8)
# # neg_itv_stats_list = interval_stats(neg_intervals, 10, 8)
# # dur_stats_list = interval_stats(durations, 10, 10)
# # res_stats_list = interval_stats(rests, 10, 10)
# # print(pos_itv_stats_list)
# # print(neg_itv_stats_list)
# # print(dur_stats_list)
# # print(res_stats_list)

# # Plot histogram
# fig, axs = plt.subplots(2, 2, sharey=False, tight_layout=True)

# n, bins, patches = axs[0][0].hist(pos_intervals, bins=50, range=(0, 100))
# axs[0][0].set_title('pos_intervals')

# n, bins, patches = axs[0][1].hist(neg_intervals, bins=50, range=(-100, 0))
# axs[0][1].set_title('neg_intervals')

# n, bins, patches = axs[1][0].hist(durations, bins=50, range=(0, 500))
# axs[1][0].set_title('durations')

# n, bins, patches = axs[1][1].hist(rests, bins=100, range=(0, 200))
# axs[1][1].set_title('rests')

plt.show()
