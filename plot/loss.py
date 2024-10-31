import sys
import matplotlib.pyplot as pl

sys.path.append("../")
with open("./output/losses_centralized.txt") as inputFile:
    loss_list = eval(inputFile.readline())

iter, loss = [], []
for _ in loss_list:
    iter.append(_[0])
    loss.append(_[1])

figure = pl.figure()
pl.plot(iter, loss, label="Test Loss", color="C4")
pl.xlabel("Iteration")
pl.ylabel("Test Loss")
pl.legend()
figure.tight_layout()
figure.show()
