#!/usr/bin/env python2

import sys

row_num = 0
col_num = -1
# all_values = []

for line in sys.stdin:
	if not line.startswith("row"):
		continue

	clean = line.replace("(","").replace(")","")
	split = clean.split()[2:]

	# read petsc style input
	#values = []
	last_index = -1
	for index, value in zip(split[0::2],split[1::2]):
		index = int(index[:-1])
		value = float(value)
		for i in range(index - (last_index+1)):
			values.append(0)
		#values.append(value)
		print value,
		last_index = index

	print

	#all_values.append(values)

	row_num += 1
	col_num = last_index+1


print '{} {}'.format(row_num, col_num)

# for line in all_values:
	#for x in line:
		#print x,
	#print
