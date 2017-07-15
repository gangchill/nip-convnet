"""
This script has to be called with the event file as argument
The resulting values can be plotted
"""
import sys
import tensorflow as tf

CEEs = []
my_tag = "CAE/cross_entropy_error"
for e in tf.train.summary_iterator(sys.argv[1]):
    for v in e.summary.value:
        if v.tag == my_tag:
            CEEs.append((e.step, v.simple_value))
print(CEEs)

MSEs = []
my_tag = "CAE/mean_squared_error"
for e in tf.train.summary_iterator(sys.argv[1]):
    for v in e.summary.value:
        if v.tag == my_tag:
            MSEs.append((e.step, v.simple_value))
print(MSEs)