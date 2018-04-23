from tasks.diagnostics.pretty.mlp import PrettyMLP
from tasks.diagnostics.pretty.rrn import PrettyRRN

rrn_4_steps = "../0ac07d0/best"
rrn_1_step = "../b7e022e/best"
mlp = "../18d93a8/best"

m = PrettyMLP()
m.load(mlp)
m.test_batches()
