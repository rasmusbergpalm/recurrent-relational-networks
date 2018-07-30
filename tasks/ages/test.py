from tasks.ages.rrn import AgesRRN
import pickle
import numpy as np

m = AgesRRN()
m.load('../138d971/best')

accs = []
while True:
    try:
        accs.append(m.test_batch())
    except Exception as e:
        print(e)
        break

with open('accs.pkl', 'wb') as fp:
    pickle.dump(accs, fp)

steps_jumps = {j: {i: [] for i in range(8)} for j in range(8)}

for acc in accs:
    for s, v in enumerate(acc):
        for j, m in v.items():
            steps_jumps[s][j].append(m)

for s in range(8):
    means = [np.mean(steps_jumps[s][j]) for j in range(8)]
    print(means)
