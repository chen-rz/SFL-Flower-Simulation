import sys
import matplotlib.pyplot as pl

sys.path.append('../')
with open("./output/metrics_centralized.txt") as inputFile:
    accu_list = eval(inputFile.readline())["accuracy"]

iter, accu = [], []
for _ in accu_list:
    iter.append(_[0])
    accu.append(_[1])

figure = pl.figure()
pl.plot(iter, accu, label="Test Accuracy", color="C3")
pl.xlabel("Iteration")
pl.ylabel("Test Accuracy")
pl.legend()
figure.tight_layout()
figure.show()
