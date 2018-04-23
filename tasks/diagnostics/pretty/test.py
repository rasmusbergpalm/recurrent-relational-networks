from tasks.diagnostics.pretty.rrn import PrettyRRN

rrn_4_steps = "../0ac07d0/best"
rrn_1_step = "../b7e022e/best"
mlp = "../18d93a8/best"

m = PrettyRRN()
m.load(rrn_4_steps)
m.test_batches()
