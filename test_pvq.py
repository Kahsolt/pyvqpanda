#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/21 

from pyvqpanda import *


# use as object
qvm = VQVM()
qvm.startup()

q = qvm.alloc_qubit()
prog = QProg() << RX(q, pi/3) << RZ(q, pi/4) << RY(q, pi/5) << H(q)
print(prog)
qvm.plot_prog(prog)

qvm.run(prog)
print(qvm.qstate)
qvm.plot_state()
qvm.plot_density()

qvm.shutdown() 


# use as context
with VQVM() as qvm:
  qvm.status()
  q = qvm.alloc_qubit(2)

  prog = QProg() << H(q[0]) << CNOT(q[0], q[1])
  print(prog)
  qvm.plot_circuit(prog, fp='img/circuit.txt')
  qvm.plot_circuit(prog, fp='img/circuit.tex')
  qvm.plot_circuit(prog, fp='img/circuit.png')

  prob = qvm.pmeasure(prog)
  print('prob:', prob)
  qvm.plot_measure(prob)
  results = qvm.measure(prog, cnt=1000)
  print('results:', results)
  qvm.plot_measure(results)

  qvm.status()
