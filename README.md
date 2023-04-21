# pyvqpanda

    A pythonic wrapper for PyQPanda's chaotic messy API...

----

**Roarrrr!! PyQPanda's API is a fucking mess, just wrap it again!!** 😫


### Quickstart

- `pip install pyqpanda`
- `python setup.py install`

```python
from pyvqpanda import *

with VQVM() as qvm:
  q = qvm.alloc_qubit(2)
  prog = QProg() << H(q[0]) << CNOT(q[0], q[1])

  #           ┌─┐
  # q_0:  |0>─┤H├ ───■──
  #           └─┘ ┌──┴─┐
  # q_1:  |0>──── ┤CNOT├
  #               └────┘
  # 
  print(prog)
  
  # >> prob: {'00': 0.5000000000000001, '01': 0.0, '10': 0.0, '11': 0.5000000000000001}
  print('prob:', qvm.pmeasure(prog))
  
  # >> result: 00
  print('result:', qvm.measure(prog, cnt=None))
  
  # >> results: {'00': 491, '11': 509}
  print('results:', qvm.measure(prog, cnt=1000))
  
  # >> qstate: [(0.7071067811865476+0j), 0j, 0j, (0.7071067811865476+0j)]
  print('qstate:', qvm.qstate)
  
  # ================
  # type: CPUQVM
  # cbits: 0
  # qubits: 2
  # qstate [(0.7071067811865476+0j), 0j, 0j, (0.7071067811865476+0j)]
  # ================
  qvm.status()
```

=> See more API examples in [test_pvq.py](test_pvq.py)


### API view

```python
class VQVM:
  @list()             # list available QVMs
  .startup()          # initialize machine
  .shutdown()         # finalize machine
  .status()           # show machine debug info
  .alloc_qubit()
  .alloc_cbit()
  .free_qubit()
  .free_cbit()
  .qstate             # current machine quantum state
  .run()              # run prog, update current machine qstate
  .pmeasure()         # theoretical probability measure
  .measure()          # repeatible Monte-Carlo measure
  .plot_circuit()     # save circuit draw
  .plot_prog()        # show single-qubit prog on bloch sphere
  .plot_state()       # show single-qubit state on bloch sphere
  .plot_density()     # show density matrix
  .plot_measure()     # show pmeasure/measure result histogram
```


#### reference

- OriginQ: [https://github.com/OriginQ](https://github.com/OriginQ)
- QPanda: [https://github.com/OriginQ/QPanda-2](https://github.com/OriginQ/QPanda-2)
  - doc: [https://pyqpanda-tutorial-en.readthedocs.io/](https://pyqpanda-tutorial-en.readthedocs.io/)
- VQNet: [https://vqnet20-tutorial.readthedocs.io](https://vqnet20-tutorial.readthedocs.io)
- Tiny-Q: [https://github.com/Kahsolt/Tiny-Q](https://github.com/Kahsolt/Tiny-Q)

----

by Armit
2023/04/21 
