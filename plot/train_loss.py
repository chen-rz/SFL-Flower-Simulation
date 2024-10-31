import sys
import matplotlib.pyplot as pl

sys.path.append('../')
with open("./output/losses_distributed.txt") as inputFile:
    loss_list = eval(inputFile.readline())

iter, loss = [], []
for _ in loss_list:
    iter.append(_[0])
    loss.append(_[1])

figure = pl.figure()
pl.plot(iter, loss, label="Train Loss", color="C2")
pl.xlabel("Iteration")
pl.ylabel("Train Loss")
pl.legend()
figure.tight_layout()
figure.show()
