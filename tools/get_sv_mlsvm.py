#!/usr/bin/env python3

import sys

sv = 0
count = 0

for line in sys.stdin:
    if "[SV][ETD], l:1" in line:
        split = line.split(":")
        sv_maj = int(split[-2].split(",")[0])
        sv_min = int(split[-1])
        print(sv_min, sv_maj)
        sv += sv_min + sv_maj
        count += 1
    elif "[RF][main] TD, l:1" in line:
        print("null")


print("last level svm {}".format(sv/float(count)))
print("count with SV {}".format(count))
