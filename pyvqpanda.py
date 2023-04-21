#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/02 

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import pyqpanda as pq
from pyqpanda import *
import pyqpanda.Visualization as pqvis
from pyqpanda.Visualization.pi_check import pi_check as round_to_pi

Cbit = ClassicalCondition
pi = np.pi
e  = np.e

class VQVM():

  ''' The virtual again universal interface for most of the `pyqpanda.QuantumMachine` :) '''

  @staticmethod
  def list():
    ''' List available QVMs '''
    for name in dir(pq):
      if name == 'QuantumMachine': continue
      obj = getattr(pq, name)
      if type(obj) == type(type) and issubclass(obj, QuantumMachine):
        print(name)

  def __init__(self, qvm_cls=CPUQVM):
    assert issubclass(qvm_cls, QuantumMachine), 'Oh, you fool got a fake QuantumMachine!'
    assert qvm_cls != QCloud, 'We do not own a QCloud!'

    self.qvm = qvm_cls()
  
  def __enter__(self) -> VQVM:
    self.startup()
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.shutdown()

  def startup(self):
    ''' Initialize machine '''
    self.qvm.init_qvm()

  def shutdown(self):
    ''' Finalize machine '''
    self.qvm.finalize()

  def status(self):
    ''' Show machine debug info '''
    print('================')
    print('type:', self.qvm.__class__.__name__)
    print('cbits:', self.qvm.get_allocate_cmem_num())
    print('qubits:', self.qvm.get_allocate_qubit_num())
    if isinstance(self.qvm, (CPUQVM, GPUQVM, CPUSingleThreadQVM)):
      print('qstate', self.qvm.get_qstate())
    #print('status', self.qvm.get_status())   # This API is broken
    print('================')

  def alloc_qubit(self, cnt=None) -> Union[Qubit, List[Qubit]]:
    return self.qvm.qAlloc() if cnt is None else self.qvm.qAlloc_many(cnt)

  def alloc_cbit(self, cnt=None) -> Union[Cbit, List[Cbit]]:
    return self.qvm.cAlloc() if cnt is None else self.qvm.cAlloc_many(cnt)

  def free_qubit(self, q:Union[Qubit, List[Qubit]]):
    if isinstance(q, Qubit): self.qvm.qFree(q)
    if isinstance(q, list):  self.qvm.qFree_all(q)

  def free_cbit(self, c:Union[Cbit, List[Cbit]]):
    if isinstance(c, Cbit): self.qvm.cFree(c)
    if isinstance(c, list): self.qvm.cFree_all(c)

  @property
  def qstate(self) -> List[complex]:
    ''' Current machine quantum state '''
    return self.qvm.get_qstate()

  def run(self, prog:QProg, noise:Noise=Noise()):
    ''' Runs a QProg, update current machine qstate '''
    self.qvm.directly_run(prog, noise_model=noise)

  def pmeasure(self, prog:QProg, q:Union[QVec, List[Qubit]]=None, type:Union[dict, tuple, list]=dict, topk:int=-1, idx:Union[int, str, List[int], List[str]]=None) \
      -> Union[Dict[str, float], List[Tuple[int, float]], List[float], complex, List[complex]]:
    '''
      PMeasure runs a QProg and returns the theoretical probability distribution or the amplitude
        - prog: the QProg to run
        - q:    the qubits to pmeasure (for full amplitude QVM), defaults to all qubits in the QProg
        - type: data type of the returned proba distro
        - topk: resturn only top-k entries
        - idx:  the qubits to pmeasure (for partial and single amplitude QVM)
      NOTE: the given QProg must NOT contain any Measure node!!
    '''

    if idx is not None:
      assert len(idx), 'index cannot be an empty list'
      if isinstance(idx, list):
        assert isinstance(self.qvm, (PartialAmpQVM, MPSQVM)), 'only PartialAmpQVM and MPSQVM support measuring partial qubits at given indexes'
        if isinstance(idx[0], int):
          return self.qvm.pmeasure_dec_subset(prog, [str(i) for i in idx])
        elif isinstance(idx[0], str):
          return self.qvm.pmeasure_bin_subset(prog, idx)
        else:
          raise TypeError(f'invalid type for idx, got {type(idx[0])}')
      elif isinstance(idx, (int, str)):
        assert isinstance(self.qvm, (SingleAmpQVM, PartialAmpQVM, MPSQVM)), 'only SingleAmpQVM, PartialAmpQVM and MPSQVM support measuring single qubit at given index'
        if isinstance(idx, int):
          return self.qvm.pmeasure_dec_index(prog, [str(i) for i in idx])
        elif isinstance(idx, str):
          return self.qvm.pmeasure_bin_index(prog, idx)
      else:
        raise TypeError(f'invalid type for idx, got {type(idx)}')
    else:
      assert isinstance(self.qvm, (CPUQVM, CPUSingleThreadQVM, GPUQVM)), 'only CPUQVM, CPUSingleThreadQVM and GPUQVM support full measuring'

      runner = {
        dict:  lambda: self.qvm.prob_run_dict,
        tuple: lambda: self.qvm.prob_run_tuple_list,     # i.e. qvm.pmeasure
        list:  lambda: self.qvm.prob_run_list,           # i.e. qvm.pmeasure_no_index
      }
      return runner[type]()(prog, q or prog.get_used_qubits([]), topk)

  def measure(self, prog:QProg, cnt:Optional[int]=1000, noise:Noise=Noise(), use_legacy:bool=False) -> Union[str, Dict[str, int]]:
    '''
      Measure runs a QProg and returns the single result or results counter
        - prog:        the QProg to run
        - cnt:         repeat times
        - noise:       quantum noise model
        - use_legacy:  use `directly_run() + quick_measure()` impl
      NOTE: the given QProg must NOT contain any Measure node!!
    '''
    '''
      Possible implementations:
        - one-shot measure
          - QProg with Measure + directly_run()
        - multi-shot measure
          - QProg with Measure + run_with_configuration()
          - QProg without Measure + directly_run() + quick_measure(); the deprecated old interface
    '''
    one_shot = cnt is None
    cnt = cnt or 1

    qubits = prog.get_used_qubits([])
    if use_legacy:
      self.qvm.directly_run(prog, noise_model=noise)
      results = self.qvm.quick_measure(qubits, shots=cnt)
    else:
      cbits = self.qvm.cAlloc_many(len(qubits))
      prog = deep_copy(prog) << measure_all(qubits, cbits)
      results = self.qvm.run_with_configuration(prog, cbits, shot=cnt, noise_model=noise)
      self.qvm.cFree_all(cbits)

    return list(results.keys())[0] if one_shot else results

  def plot_circuit(self, prog:QProg, fp:str='circuit.txt', clock:bool=False):
    ''' Export and save circuit draw '''
    suffix = Path(fp).suffix.lower()
    if suffix == '.txt':
      if clock: draw_qprog_text_with_clock(prog, output_file=fp)
      else:     draw_qprog_text           (prog, output_file=fp)
    elif suffix == '.tex':
      if clock: draw_qprog_latex_with_clock(prog, output_file=fp)
      else:     draw_qprog_latex           (prog, output_file=fp)
    elif suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']:
      pqvis.circuit_draw.draw_circuit_pic(prog, fp)
    else:
      raise ValueError(f'unspported filetype: {suffix}')

  def plot_prog(self, prog:QProg, fp:str=None):
    ''' Show animation of single-qubit circuit on bloch sphere '''
    assert len(prog.get_used_qubits([])) == 1, 'only support single-qubit circut'
    pqvis.plot_bloch_circuit(prog, saveas=fp)

  def plot_state(self, state:np.ndarray=None):
    ''' Show single-qubit state on bloch sphere '''
    state = state or self.qstate
    if len(state) > 2: print('entangled multi-qubits might not work properly, take caution')
    pqvis.plot_bloch_multivector(state)

  def plot_density(self, state:np.ndarray=None):
    ''' Plot density matrix of quantum state / vector '''
    state = state or self.qstate
    if len(state):
      pqvis.plot_state_city(state, title='real/imag of rho')
      #pqvis.plot_density_matrix(pqvis.state_to_density_matrix(state))
    else:
      print('Empty qstate, please run a program first!')

  def plot_measure(self, results:Dict[str, Union[float, int]], ignore_zero=False):
    ''' Plot histograms of pmeasure / measure results '''
    if ignore_zero: pqvis.draw_probaility_dict(results)
    else:           pqvis.draw_probaility(results)
    plt.show()

if 'control':
  QIf    = QIfProg
  QWhile = QWhileProg

if not 'variational':
  VQC      = VariationalQuantumCircuit
  vI       = VariationalQuantumGate_I
  vX       = VariationalQuantumGate_X
  vY       = VariationalQuantumGate_Y
  vZ       = VariationalQuantumGate_Z
  vH       = VariationalQuantumGate_H
  vX1      = VariationalQuantumGate_X1
  vY1      = VariationalQuantumGate_Y1
  vZ1      = VariationalQuantumGate_Z1
  vRX      = VariationalQuantumGate_RX
  vRY      = VariationalQuantumGate_RY
  vRZ      = VariationalQuantumGate_RZ
  vS       = VariationalQuantumGate_S
  vT       = VariationalQuantumGate_T
  vU1      = VariationalQuantumGate_U1
  vU2      = VariationalQuantumGate_U2
  vU3      = VariationalQuantumGate_U3
  vU4      = VariationalQuantumGate_U4
  vSWAP    = VariationalQuantumGate_SWAP
  viSWAP   = VariationalQuantumGate_iSWAP
  vSqiSWAP = VariationalQuantumGate_SqiSWAP
  vCNOT    = VariationalQuantumGate_CNOT
  vCR      = VariationalQuantumGate_CR
  vCZ      = VariationalQuantumGate_CZ
  vCU      = VariationalQuantumGate_CU
  vCRX     = VariationalQuantumGate_CRX
  vCRY     = VariationalQuantumGate_CRY
  vCRZ     = VariationalQuantumGate_CRZ
