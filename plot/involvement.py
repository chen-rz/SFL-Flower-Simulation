import sys
import matplotlib.pyplot as pl

sys.path.append('../')
with open("./output/involvement_history.txt") as inputFile:
    involvement_history = eval(inputFile.readline())

cid = range(len(involvement_history))

figure = pl.figure()
pl.bar(cid, involvement_history, label="Involvements", color="C5")
pl.xlabel("Device ID")
pl.ylabel("Involvement")
pl.legend()
figure.tight_layout()
figure.show()
