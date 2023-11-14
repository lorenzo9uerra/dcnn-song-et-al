import numpy as np
import ast
import sys

with open(sys.argv[1], "r") as data:
    results= ast.literal_eval(data.read())

print("seed:",results.pop('seed'))
for attack in results.keys():
    print(attack)
    print(f"\n\tPrecision average: {np.mean(results[attack]['precisions'])} std: {np.std(results[attack]['precisions'])}")
    print(f"\tRecall average: {np.mean(results[attack]['recalls'])} std: {np.std(results[attack]['recalls'])}")
    print(f"\tF1-score average: {np.mean(results[attack]['f1-scores'])} std: {np.std(results[attack]['f1-scores'])}\n")
