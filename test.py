import numpy
import json
import matplotlib.pyplot as plt
import csv

data1 = []
data2 = []
data3 = []
f = open('data.csv')
csvreader = csv.reader(f, delimiter=";")

rows = []
for row in csvreader:
    data1 += [float(row[0])]
    data2 += [float(row[1])]
    data3 += [float(row[2])]

x = [i+1 for i in range(0, len(data1))]

markers = ["v", "o", "x", "s", "*", ".", "+", "s", "d"]
scales = [20, 70, 20, 20, 70, 20]

plt.plot(x, data1, label="NTS")
plt.plot(x, data2, label="FTS")
plt.plot(x, data3, label="PTS")

plt.rcParams.update({'font.size': 20})

# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.legend(bbox_to_anchor=(0.95, 0.4), loc='upper right')
plt.tight_layout()
plt.xlabel("Training iteration", size=15)
plt.ylabel("Mean episode reward", size=15)
plt.show()
