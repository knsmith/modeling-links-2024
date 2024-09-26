import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import functools
import sys

from copy import deepcopy

import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit

from collections.abc import Iterable

from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.measure import Measure
from qiskit.circuit import Delay
from qiskit.circuit.library.standard_gates.s import SdgGate
from qiskit.circuit.library.standard_gates.s import SGate
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.library.standard_gates.swap import SwapGate
from qiskit.circuit.library import IGate
    
def size_breakdown(c):
    length = None
    try:
        length = len(c)
        sub_sizes = [size_breakdown(x) for x in c]
        sub_sizes_pruned = [s for s in sub_sizes if s is not None]
        if len(sub_sizes_pruned == 0):
            return length
        return [length, sub_sizes_pruned]
    except TypeError:
        pass
    return None

def sparsify_counts_dicts(ds):
    """
    Makes a counts sparse without loading the dictionaries.
    """

def get_key_length(d):
    key_length = None
    for key in d:
        key_length = len([bit for bit in key.strip() if ((bit == '0') or (bit == '1'))])
        break
    return key_length

def split_dict_by_n_qubits(d, n):
    """
    Given a dictionary of counts, creates individuals for clusters of adjacent n qubits
    """
    assert(isinstance(d, dict))
    
    key_length = get_key_length(d)
    assert((key_length % n) == 0)
    
    new_dictionaries = tuple(dict() for i in range(key_length // n))
    
    for key,v in d.items():
        current = ''.join(key.strip().split(' '))
        for i in range(0, key_length, n):
            iovern = i // n
            bits = current[i:i+n]
            new_dictionaries[iovern][bits] = new_dictionaries[iovern].get(bits, 0) + v
    
    #The -1 might be needed for little-endian order
    return new_dictionaries[::-1]
    

def split_dict_by_qubit(d):
    """
    Given a dictionary of counts, creates individual dictionaries for separate qubits.
    
    MIGHT BE DEPRECATED TO INSTEAD USE ABOVE SOON
    """
    assert(isinstance(d, dict))
    
    key_length = get_key_length(d)
    
    new_dictionaries = tuple(dict([['0',0],['1',0]]) for i in range(key_length))
    
    for key,v in d.items():
        for i, bit in enumerate(key.strip().split(' ')):
            new_dictionaries[i][bit] += v
    
    return new_dictionaries[::-1]

def bunch_pairs(index_list):
    """
    Takes a list of sortable items. Assumes adjacent ones should form pairs. Makes pairs this way.
    """
    #assert((len(index_list) % 2) == 0)
    if (len(index_list) % 2) == 0:
        return [frozenset([index_list[i], index_list[i+1]]) for i in range(0, len(index_list), 2)]
    else:
        return [frozenset([index_list[i], index_list[i+1]]) for i in range(0, len(index_list)-1, 2)]

def read_prop_dict(fconf_name):
    bprops = None
    with open(fconf_name) as f1:
        local_temp = {}
        exec("import datetime\ntzlocal = lambda : None\n\n" + "bprop_d = " + f1.read(), local_temp, local_temp)
        bprops = local_temp['bprop_d']
    
    return bprops
    
def read_counts_dicts(fdata_name):
    with open(fdata_name) as f2:
        str1 = f2.read()
        dict_strings = str1[2:-2].split('}, {')
        del str1
        #print("Split length: %d" % len(dict_strings))
        data = list((eval('{' + dstr + '}') for dstr in dict_strings))
        del dict_strings
    for dat in data:
        assert(isinstance(dat, dict))
    
    return data

remote_backend_res_dict = dict(backend_name="remote_backend",
    backend_version="1.0.0",
    qobj_id="id-123",
    job_id="job-123",
    success=True)

def reassemble_result(counts, circs=None):
    """
    Reconstructs a results object from counts.
        counts - a list of dicts, each corresponding to an experiment
    """
    from qiskit.result import Result
    from qiskit.result.models import ExperimentResultData, ExperimentResult
    from qiskit.qobj import QobjExperimentHeader
    
    
    exp_res_list = []
    for i, counts_exp in enumerate(counts):
        a_key = next(iter(counts_exp.keys()))
        creg_sizes = [[("c%d" % i), s] for i, s in enumerate(reversed([len(k) for k in a_key.split(' ')]))]
        creg_total = sum((item[1] for item in creg_sizes), 0)
        
        if circs is not None:
            name = circs[i].name
            #print(name)
        else:
            name = None
        data = ExperimentResultData(counts_exp)
        shots = sum(counts_exp.values())
        exp_result_header = QobjExperimentHeader(
            creg_sizes = creg_sizes, memory_slots=creg_total, name=name)
        exp_result = ExperimentResult(
            shots=shots, success=True, meas_level=2, data=data, header=exp_result_header)
        exp_res_list.append(exp_result)
    res = Result(results=exp_res_list, **remote_backend_res_dict)
    res_counts = res.get_counts()
    #print(counts)
    #print(res_counts)
    #assert(res_counts == counts)
    return res

def merge_q_registers(circ, qubit_number):
    """
    Given a circuit that has been defined on individually-named quantum registers of decimal format q1...qN, converts to a monolithic register 
    
    circ - a circuit on many registers
    qubit_number - how many qubits total should be in the monolithic system
    
    returns a quantum circuit, merged
    """
    qreg_mono = QuantumRegister(qubit_number)
    classical_bits = circ.clbits
    cl_regs = frozenset(cb.register for cb in classical_bits)
    
    def parse_name(qreg_name):
        return int(qreg_name[1:])
    
    qc = QuantumCircuit(*([qreg_mono] + list(cl_regs)))
    #Now iterate the item, see which qubits each thing touches
    measured_qubits = []
    for item in circ:
        #print(item)
        item_qubits= item[1]
        item_cbits = item[2]
        
        #print(item_qubits)
        #print([qubit.register.name for qubit in item_qubits])
        qubit_indices = [parse_name(qubit.register.name) for qubit in item_qubits]
        qubits_new = [qreg_mono[index] for index in qubit_indices]
        #Keep the same classical bits & registers
        
        if isinstance(item[0], HGate):
            qc.h(*qubits_new)
        elif isinstance(item[0], XGate):
            qc.x(*qubits_new)
        elif isinstance(item[0], Barrier):
            qc.barrier(*qubits_new)
        elif isinstance(item[0], Delay):
            qc.delay(duration=item[0].duration, qarg=qubits_new, unit=item[0].unit)
        elif isinstance(item[0], SdgGate):
            qc.sdg(*qubits_new)
        elif isinstance(item[0], SGate):
            qc.s(*qubits_new)
        elif isinstance(item[0], CXGate):
            qc.cx(*qubits_new)
        elif isinstance(item[0], SwapGate):
            qc.swap(*qubits_new)
        elif isinstance(item[0], IGate):
            qc.id(*qubits_new)
        elif isinstance(item[0], Measure):
            qc.measure(qubits_new[0], item_cbits[0])
            measured_qubits.extend(qubit_indices)
        else:
            raise ValueError("Gate %s of unsupported type %s" % (str(item), item[0].__class__.__name__))
    
    return [qc, measured_qubits]

def filter_circuits_reject_basis_changes(circuits):
    """
    Given a collection of circuits, finds those having only X gates, which are useful for readout error mitigation
    """
    to_return = []
    for i,circ in enumerate(circuits):
        flag = True
        for item in circ:
            if isinstance(item[0], HGate):
                flag = False
                break
            elif isinstance(item[0], Delay):
                flag = False
                break
            elif isinstance(item[0], SdgGate):
                flag = False
                break
            elif isinstance(item[0], SGate):
                flag = False
                break
            elif isinstance(item[0], CXGate):
                flag = False
                break
            elif isinstance(item[0], SwapGate):
                flag = False
                break
            elif isinstance(item[0], IGate):
                flag = False
                break
        if flag:
            to_return.append([i,circ])
    return to_return

def shallow_flatten(l):
    return [item for sublist in l for item in sublist]
        
def deep_flatten(l):
    #from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes, QuantumCircuit)):
            yield from deep_flatten(el)
        else:
            yield el
#end of method

def extract_from_entry(entry, name, key, entrykey):
    """
    Used to extract fields from entries in properties' dictionary-list structures
    """
    for item in entry:
        if item[name] == key:
            return item[entrykey]
    raise KeyError("Could not find %s:%s in entry %s" % (name, key, str(entry)))
#end of method

def add_to_adjacent(rows, qubit, link_dict):
    """
    Adds qubit to the row to which it is linked
    Qubit must appear at the end of the row
    """
    linked = link_dict(qubit)
    for row in rows:
        for lbit in linked:
            if row[0] == lbit:
                row.insert(0, qubit)
                return True
            if row[-1] == lbit:
                row.append(qbit)
                return True
    return False
            
def circ_sum(circs):
    """
    Conveniene method to create a sum of quantum circuits.
    """
    return sum(circs, QuantumCircuit())

def trim_count_key(k):
    return "".join(k.split())

def take_some_bits(s, n, end=False):
    """
    Takes 1st n bits from string s, assuming they are separated by spaces.
    """
    i = 0
    bits = []
    while len(bits) < n:
        if end:
            if s[-i-1] == '0' or s[-i-1] == '1':
                bits.append(s[-i-1])
        else:
            if s[i] == '0' or s[i] == '1':
                bits.append(s[i])
        i += 1
    if end:
        return [''.join(bits), s[:-i-1]]
    return [''.join(bits), s[i:]]

def trim_count_keys(counts):
    """
    Eliminiates spaces from a counts dictionary
    """
    return ((trim_count_key(k), v) for k,v in counts.items())

def order_best(pair, readout_errs):
    return [k for k,err in sorted([[k, readout_errs[k]] for k in pair], key=lambda x : x[1])]

def all_equal(s):
    return functools.reduce(lambda acc, x : acc and x, (c == s[0] for c in s), True)

def charwise_match(k):
    regs = k.split(' ')
    reg_length = len(regs[0])
    for reg in regs[1:]:
        assert(reg_length == len(reg))
    return [all_equal([reg[i] for reg in regs]) for i in range(reg_length)]

def bitwise_maj(k):
    regs = k.split(' ')
    choices1 = (-1,1)
    choices2 = ('0','1')
    reg_length = len(regs[0])
    for reg in regs[1:]:
        assert(reg_length == len(reg))
    totals = ((sum(choices1[reg[i] == '1'] for reg in regs) > 0) for i in range(reg_length))
    return ''.join(choices2[t > 0] for t in totals)

def post_select(counts, par_cregs_num, post_at_beginning = False, invert=False):
    """
    Implements post-selection on measurements being 0
    
    Can do correction after this method.
    """
    if par_cregs_num == 0:
        return [counts, {}]
    
    trimmed = ([take_some_bits(k, par_cregs_num, not post_at_beginning), v] for k,v in counts.items())
    
    if post_at_beginning:
        process_counts = ([(k1, k2), v] for (k1,k2),v in trimmed)
    else:
        process_counts = ([(k1, k2), v] for (k1,k2),v in trimmed)
    
    if invert:
        target_string = '1'*par_cregs_num
    else:
        target_string = '0'*par_cregs_num
    
    equals_target = [[k,v,k[0] == target_string] for k,v in process_counts]
    #print(list(equals_target))
    
    valid_counts = dict((k[1],v) for k,v,eq in equals_target if eq)
    other_counts = dict((k,v)  for k,v,eq in equals_target if not eq)
    
    return [valid_counts, other_counts]

def list_to_chunks(l, chunklengths):
    totalread = 0
    all_chunks = []
    for cl in chunklengths:
        all_chunks.append(l[totalread:totalread+cl])
        totalread += cl
    return all_chunks

def expand_from_start(start_pt, upto):
    #print(start_pt)
    positive = True
    curr_left = 0
    curr_right = 0
    row = []
    for i in range(upto-1):
        if positive:
            row.append([start_pt + curr_right, start_pt + curr_right + 1])
            curr_right += 1
        else:
            row.append([start_pt - curr_left, start_pt - curr_left - 1])
            curr_left += 1
        positive = not positive
    #print(row)
    #print()
    return row

def plot_readable(title, ylabel, xlabel, x_data, y_data, y_labels = None,
                  colors = None, styles = ('-', '--', '-.', ':')):
    plt.figure(figsize=(12,10))
    #plt.title(title, fontsize=32)
    plt.ylabel(ylabel,fontsize=32)
    plt.xlabel(xlabel,fontsize=32)
    y_list = []
    if colors is None:
        colors = [None]*len(x_data)
    for i, (xd, yd) in enumerate(zip(x_data, y_data)):
        line, = plt.plot(xd, yd, color=colors[i], linestyle = styles[i % len(styles)])
        y_list.append(line)
    if y_labels:
        plt.legend(y_list, y_labels, fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=32)
    plt.show()
    return