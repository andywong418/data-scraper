from __future__ import division
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import json
import matplotlib.pyplot as plt

graph_plot = []
with open('input_graph_progress_intelligent_4.json', 'r') as outfile:
    json_data = json.load(outfile)
    samples_axis = json_data['samples_axis']
    accuracy_axis = json_data['accuracy_axis']
    graph_plot.append([samples_axis, accuracy_axis])

with open('input_graph_progress_random.json', 'r') as outfile:
    json_data = json.load(outfile)
    samples_axis = json_data['samples_axis'][0:65]
    accuracy_axis_1 = json_data['accuracy_axis'][0:65]
    graph_plot.append([samples_axis, accuracy_axis_1])
with open('input_graph_progress_sift.json', 'r') as outfile:
    json_data = json.load(outfile)
    samples_axis = json_data['samples_axis']
    accuracy_axis_2 = json_data['accuracy_axis']
    graph_plot.append([samples_axis, accuracy_axis_2])

# for index, double in enumerate(graph_plot):
#   print("index", index)
#   if index == 0:
#       plt.plot(double[0], double[1], 'r-', label='SURF+SVM selection')
#   # elif index == 1:
#   #     plt.plot(double[0], double[1], label='Random Selection')
#   else:
#       plt.plot(double[0], double[1], label='Random Selection')
#
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=2, mode="expand", borderaxespad=0.)
#
# plt.savefig('graph_plot_good_1.png')

accuracy_axis = np.array(accuracy_axis_1)
accuracy_axis_2 = np.array(accuracy_axis_2[0:65])
difference = np.subtract(accuracy_axis_2, accuracy_axis_1)
print (np.mean(difference) * 100)
# #
# # array = [2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 1, 2, 1, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 0, 1, 2, 1, 1, 1, 0, 1, 0, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 0, 1, 2, 1, 0, 0, 0, 2, 2, 2, 2, 1, 1, 0, 1, 2, 2, 0, 0, 0, 0, 2, 0, 1, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 0, 1, 2, 2, 2, 1, 2, 1, 2, 1, 0, 1, 2, 2, 0, 1, 2, 2, 2, 1, 0, 1, 2, 2, 1, 0, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 0, 1, 2, 1, 1, 2, 1, 0, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 0, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 0, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 0, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 0, 2, 2, 0, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 0, 2, 0, 2, 2, 0, 0, 1, 1, 0, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 0, 2, 1, 2, 2, 2, 2, 0, 2, 0, 1, 2, 0, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 0, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 0, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 2, 0, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2]
#
# counter = 0
# for number in array:
#     if number == 2:
#         counter += 1
# print(len(array))
# print("acc", counter/len(array))
