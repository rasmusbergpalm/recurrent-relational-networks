from tasks.diagnostics.pretty.rrn import PrettyRRN

rrn_4_steps = "../0cf2607/best"
rrn_1_step = "../b7e022e/best"
mlp = "../8a972eb/best"

m = PrettyRRN()
m.load(rrn_4_steps)
m.test_batches()
