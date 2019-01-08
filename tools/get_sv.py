#!/usr/bin/env python3

import sys

next_best = False
sv = 0
count = 0
last_line = ""

for line in sys.stdin:
    if "refinement at level 0" in line:
        split = last_line.split(":")
        sv_min = int(split[10].split(" ")[0])
        sv_maj = int(split[11].split(" ")[0])
        print(sv_min, sv_maj)
        sv += sv_min + sv_maj
        count += 1

    last_line = line


print("last level svm {}".format(sv/float(count)))
