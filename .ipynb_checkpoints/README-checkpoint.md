# pseudo_multicore_processor

This repo contains a pseudo multi-core processor simulator implementation in Python. The simulator is designed to process simple code and simulate its execution on a specified number of PEs.
The simulator is very basic and can only perform simple arithmetic operations. 
## Project Input/Output

### Input Code
The parser expects code in a .txt file as input with the following requirements:
- only these arithmetic operations can be performed: “+, -, *,  /”.
- no branches, loops etc
- final output must be stored in the variable 'y'

An example is shown below. In addition, codeex.txt file is provided as an example.

```python
x = 5
t1 = x
t2 = t1 + 4
t3 = t1 * 8
t4 = t1 - 4
t5 = t1 / 2
t6 = t2 * t3
t7 = t4 - t5
t8 = t6 * t7
y = t8
```
### Output

Sample output for the code in codeex.txt, PEs=3.

```console
Placed <Instruction(('x', '5'))> at cycle 0 on PE 0
Placed <Instruction(('t1', 'x'))> at cycle 1 on PE 0
Placed <Instruction(('t2', 't1', '+', '4'))> at cycle 2 on PE 0
Placed <Instruction(('t3', 't1', '*', '8'))> at cycle 2 on PE 1
Placed <Instruction(('t5', 't1', '/', '2'))> at cycle 2 on PE 2
Placed <Instruction(('t4', 't1', '-', '4'))> at cycle 3 on PE 0
Placed <Instruction(('t6', 't2', '*', 't3'))> at cycle 4 on PE 0
Placed <Instruction(('t7', 't4', '-', 't5'))> at cycle 11 on PE 0
Placed <Instruction(('t8', 't6', '*', 't7'))> at cycle 12 on PE 0
Placed <Instruction(('y', 't8'))> at cycle 14 on PE 0
Placed <Instruction(('NOP',))> at cycle 6 on PE 0
Placed <Instruction(('NOP',))> at cycle 7 on PE 0
Placed <Instruction(('NOP',))> at cycle 8 on PE 0
Placed <Instruction(('NOP',))> at cycle 9 on PE 0
Placed <Instruction(('NOP',))> at cycle 10 on PE 0
Placed <Instruction(('NOP',))> at cycle 0 on PE 1
Placed <Instruction(('NOP',))> at cycle 1 on PE 1
Placed <Instruction(('NOP',))> at cycle 0 on PE 2
Placed <Instruction(('NOP',))> at cycle 1 on PE 2
PE 0 [<Instruction(('x', '5'))>, <Instruction(('t1', 'x'))>, <Instruction(('t2', 't1', '+', '4'))>, <Instruction(('t4', 't1', '-', '4'))>, <Instruction(('t6', 't2', '*', 't3'))>, <Instruction(('NOP',))>, <Instruction(('NOP',))>, <Instruction(('NOP',))>, <Instruction(('NOP',))>, <Instruction(('NOP',))>, <Instruction(('t7', 't4', '-', 't5'))>, <Instruction(('t8', 't6', '*', 't7'))>, <Instruction(('y', 't8'))>]
PE 1 [<Instruction(('NOP',))>, <Instruction(('NOP',))>, <Instruction(('t3', 't1', '*', '8'))>]
PE 2 [<Instruction(('NOP',))>, <Instruction(('NOP',))>, <Instruction(('t5', 't1', '/', '2'))>]
Running with 3 PEs took 15 cycles
{'x': 5, None: None, 't1': 5, 't2': 9, 't4': 1, 't3': 40, 't6': 360, 't5': 2.5, 't7': -1.5, 't8': -540.0, 'y': -540.0}
```
Intermediate Representation and the DAG visualization will be dumped in the output directory.

## Install and run project

### Requirements 

Required libraries:
- [regex](https://pypi.org/project/regex/)
- [graphviz](https://pypi.org/project/graphviz/)
```console
pip install -r requirements.txt 
```

### Run Project

```console
python pseudo_processor.py --path path/to/codetxtfile --pe NUM_PEs --pl naive
```
Note: greedy placement implementation is available, however does not work for all cases. Running naive placement is recommended. 
