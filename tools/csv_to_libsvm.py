#!/usr/bin/env python2

import sys

label = ""

if len(sys.argv) == 2:
    label = sys.argv[1]

firstline = True
for line in sys.stdin:
    if firstline:
        firstline = False
        continue

    processedlabel = False

    if label:
        print label,
        processedlabel = True

    idx = 1
    for word in line.split(","):
        if not processedlabel:
            print word,
            processedlabel = True
            continue

        if word[-1] == '\n':
            word = word[:-1]

        print "{}:{}".format(idx, word),
        idx += 1
    print
