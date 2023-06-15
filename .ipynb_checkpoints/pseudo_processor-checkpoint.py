import argparse
import os 
import re

from collections import OrderedDict, deque, defaultdict

from graphviz import Source


class DAG():
    """
    Directed acyclic graph implementation.
    Implemented from https://github.com/thieman/py-dag.
    """
    def __init__(self):
        self.graph = OrderedDict()

    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = set()

    def add_edge(self, from_node, to_node):
        if from_node not in self.graph:
            self.add_node(from_node)
        if to_node not in self.graph:
            self.add_node(to_node)
        self.graph[from_node].add(to_node)
        try:
            self.topological_sort()
        except ValueError:
            self.graph[from_node].remove(to_node)
            raise ValueError(f"Adding {from_node} -> {to_node} causes a cycle")

    def predecessors(self, node):
        return [key for key in self.graph if node in self.graph[key]]

    def downstream(self, node):
        if node not in self.graph:
            raise KeyError(f"{node} is not in graph")
        return list(self.graph[node])

    def leaves(self):
        return [key for key in self.graph if not self.graph[key]]

    def independent_nodes(self):
        """
        All nodes that nobody depends on. Our starting points.
        """
        dependent_nodes = set(
            node for dependents in self.graph.values() for node in dependents
        )
        return [node for node in self.graph.keys() if node not in dependent_nodes]

    def topological_sort(self):
        in_degree = {}
        for u in self.graph:
            in_degree[u] = 0

        for u in self.graph:
            for v in self.graph[u]:
                in_degree[v] += 1

        queue = deque()
        for u in in_degree:
            if in_degree[u] == 0:
                queue.appendleft(u)

        out = []
        while queue:
            u = queue.pop()
            out.append(u)
            for v in self.graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.appendleft(v)
        if len(out) == len(self.graph):
            return out
        raise ValueError("Graph is not acyclic")

    def get_dot(self):
        ret = "digraph dag {\n"
        for u in self.graph:
            ret += f"  \"{u}\" -> {{\"" + "\" \"".join(str(v) for v in self.graph[u]) + "\"};\n"
        ret += "}"
        return ret



class Instruction:
    """
    Takes in raw instruction extracted by regex from .txt file, eg ('t2', 't1', '+', '4').
    Outputs an Instruction object that contains all relevant information about the instruction:
    
    - raw instruction
    - instrruction name
    - all variables in the instruction
    - instruction cost, ie num cycles to execute.
    
    """
    def __init__(self, raw_instr):
        
        # Number of cycles it takes to complete each instruction
        INSTRUCTION_COSTS = {
            "LOAD": 1,
            "NOP": 1,
            "*": 2,
            "-": 1,
            "+": 1,
            "/": 8,
        }
        
        var_re = re.compile(r'([a-z]{1}[a-z0-9]*)')
        
        self.raw = raw_instr
        self.name = raw_instr[0]
        self.variables = [x for x in raw_instr[1:] if var_re.match(x)]
        self.cost = INSTRUCTION_COSTS[self._get_instr()]

    def _get_instr(self):
        if len(self.raw) == 4:
            return self.raw[2]
        elif len(self.raw) == 2:
            return "LOAD"
        elif len(self.raw) == 1:
            return "NOP"
        
    def __str__(self):
        return f"<Instruction({self.raw})>"
    
    def __repr__(self):
        return str(self)



class ProcessingElement:
    """
    This class represents a single processing element (PE).
    Takes in the PE index and outputs a ProcessingElement object that contains 
    instructions in execution order on this specific PE.
    
    """
    def __init__(self, idx):
        # Instructions in execution order
        self.instructions = []
        # PE index
        self.idx = idx

    def current_cost(self):
        """
        Calculates the max possible number of cycles for a single operation.
        
        """
        return max(start + instr.cost for start, instr in self.instructions)

    def insert_instruction(self, time, instr):
        """
        Finds an open interval during which an instruction can be be executed and inserts 
        instruction in the appropriate position.
        
        Inputs:
        instr - new instruction that must be placed;
        time - min time at which the new instruction can start execution.
        
        Outputs:
        bool
        
        """
        # New instr start time
        ns = time
        # New instr end time
        ne = time + instr.cost
        
        # Loop though existing instructions and find an open placement interval
        for es, existing_instr in self.instructions:
            # Existing instr end time
            ee = es + existing_instr.cost
            # If overlap, cannot insert
            if not (ne <= es or ee <= ns):
                return False
            
        # If no overlap, free interval is available, insert instr
        self.instructions.append((time, instr))
        # Sort instructions by start time
        self.instructions.sort(key=lambda x: x[0])
        print(f"Placed {instr} at cycle {time} on PE {self.idx}")
        return True

    def fill_nops(self):
        """
        Load balancing function that adds "NOPs" to balance the running times.
        
        """
        # Max possible num of cycles
        end = self.current_cost()
        
        # Insert NOPs if needed to match max num cycles
        for i in range(end):
            self.insert_instruction(i, Instruction(("NOP",)))
  
    def list_instructions(self):
        """
        Outputs list of instructions.
        
        """
        return [instr for _, instr in self.instructions]



class PseudoBackend:
    """
    This class represents the backend - places the instructions accross all PEs in execution order.
    
    """
    
    def __init__(self, IR, PEs, placement="naive"):
        # List of available PEs
        self.processors = [ProcessingElement(idx) for idx in  range(PEs)]
        # IR (the DAG we created earlier)
        self.IR = IR

        # Calculate min time node has to wait before it can be scheduled on any PE
        # Store all min costs (wait times) in a dict
        node_min_wait = defaultdict(int)
        nodes = G.independent_nodes()
        
        for node in nodes:
            partial_cumul_costs = self.calc_cost_rec(node, 0)
            for cost in partial_cumul_costs:
                node_min_wait[cost[0]] = max(node_min_wait[cost[0]], cost[1])
                
        # Place nodes (instructions) on PEs
        if placement == "greedy":
            self.greedy_placement(node_min_wait)
        else:
            self.naive_placement(node_min_wait)

        for pe in self.processors:
            pe.fill_nops()

        for pe in self.processors:
            print("PE", pe.idx, pe.list_instructions())

    def calc_cost_rec(self, node, current_cost):
        """
        Recursively calculates the cumulative cost of the input node (DFS), ie how long
        it takes to execute this node considering prev operations it is dependent on.

        """
        out = [(node.name, current_cost)]
        cost = current_cost + node.cost
        
        for dependant in self.IR.downstream(node):
            out += self.calc_cost_rec(dependant, cost)
        return out


    def naive_placement(self, min_waits):
        """
        Naively places instructions based on PE availability (first come, first serve).
        """
        # Topological sort - returns a list with instructions in dependency order
        instructions = self.IR.topological_sort()
        
        # The placement loop below relies on min number of cycles instruction must wait before 
        # it can run. Since this method is naive, other instructions may have already been placed
        # on other PEs, which may offset the min wait time by a specific num of cycles.
        # The offsets are stored in a dict and used to adjust the wait times.
        offsets = defaultdict(int)
        
        for instr in instructions:
            # Get the min possible cumulative instruction cost
            min_wait = min_waits[instr.name]
            not_placed = True
            
            # Find a placement for the instruction
            while not_placed:
                # loop though all available PEs
                for pe in self.processors:
                    # Based on already adjusted dependencies, select the worst case and schedule with that offset
                    offset = max([offsets[var] for var in instr.variables] + [0])
                    # If a slot is available on a PE, place the instr and break the loop
                    if pe.insert_instruction(min_wait + offset, instr):
                        not_placed = False
                        offsets[instr.name] = sum([offsets[instr.name], offset])
                        break
                # If none of the PEs are available, increment cumul instr cost, offset and try again        
                if not_placed:
                    min_wait += 1
                    offsets[instr.name] += 1
                    
            
    def greedy_placement(self, min_waits):
        """
        Greedy instruction placement. Implememtation very similar to naive, 
        but instructions with highest cost are placed first.
        """
        
        instr_with_cost = []
        instructions = self.IR.topological_sort()
        
        # Create a lits of tuples (instruction cost, Instruction)
        for instr in instructions:
            instr_with_cost.append((instr.cost, instr))
        
        # Sort instructions by cost in high --> low order
        instr_with_cost.sort(key=lambda x: x[0], reverse=True)
        
        offsets = defaultdict(int)
        for _, instr in instr_with_cost:
            min_wait = min_waits[instr.name]
            not_placed = True
            
            while not_placed:
                for pe in self.processors:
                    offset = max([offsets[var] for var in instr.variables] + [0])
                    if pe.insert_instruction(min_wait + offset, instr):
                        not_placed = False
                        offsets[instr.name] = sum([offsets[instr.name], offset])
                        break
                        
                if not_placed:
                    min_wait += 1
                    offsets[instr.name] += 1




class Runtime:
    """
    This class executes the backend code on all PEs and returns the arithmetic results.
    Takes as input PEs with instructions from the backend and runs all codes. 
    
    """
    def __init__(self, pe_instructions):
        self.pe_instructions = pe_instructions
        self.current_instructions = {}
        self.current_instruction_ends = {}
        self.memory = {}
        self.cycle = 0

    def run(self):
        """
        Runs and executes code on all PEs.
        """
        # Run instructions on each cycle. One iteration of the loop
        # represents 1 cycle. 
        while True:
            res = []
            
            # Process instructions an all PEs
            for pe_idx in range(len(self.pe_instructions)):
                res += self._process(pe_idx, self.cycle, self.memory)
                
            # Store results for each variable in memory.
            # if instruction was fully executed, result will be available in the list
            for var, val in res:
                self.memory[var] = val
                
            self.cycle += 1
            
            # 'y' is the final instruction 
            if 'y' in self.memory:
                break
            


    def _process(self, pe_idx, cycle, memory):
        """
        Executes code on a specific PE and returns its output. 
        """
        out = []
        
        # Get PE instructions and their start times 
        instructions = self.pe_instructions[pe_idx]
        for start, instr in instructions:
            # Current cycle matches instruction start time 
            # Add instruction its end time to dicts 
            if start == cycle:
                self.current_instructions[pe_idx] = instr
                self.current_instruction_ends[pe_idx] = start + instr.cost - 1
                
                # Check if all variables in instruction have been loaded into memory
                # If not, raise exception
                for var in instr.variables:
                    if var not in memory:
                        raise Exception(f"Failed to read memory for {var}, missing variable in cycle {cycle}, was executing {instr} on PE {pe_idx}")
        
        # Simulate instruction execution and compute the result
        if self.current_instructions[pe_idx]:
            if self.current_instruction_ends[pe_idx] == cycle:
                out += [self._execute(self.current_instructions[pe_idx], memory)]
                # Remove instruction from the PE key in dict
                self.current_instructions[pe_idx] = None
        return out
  
    def _execute(self, instr, memory):
        """
        Executes specific instruction and returns the numerical output in format (var, result)
        """
        raw = instr.raw
        
        # Returns integer value of a constant or variable form memory
        def var_or_const(val):
            if val.isdigit():
                # constant
                return int(val)
            else:
                # load from memory
                return memory[val]

        # NOP
        if len(raw) == 1:
            return (None,None)

        # LOAD
        if len(raw) == 2:
            return (raw[0], var_or_const(raw[1]))

        # ARITHMETIC (Compute result)
        ops = {
          "+": lambda a, b: a + b,
          "-": lambda a, b: a - b,
          "*": lambda a, b: a * b,
          "/": lambda a, b: a / b,
        }
        # Perform arithmetic operation
        if len(raw) == 4:
            # Get operation 
            op = raw[2]
            # Load numerical values
            a = var_or_const(raw[1])
            b = var_or_const(raw[3])
            if op not in ops:
                raise Exception(f"Unsupported math op {op}")
                
            # Perform arithmetic operation and return result 
            return (raw[0], ops[op](a, b))




def simulator(IR, PEs, placement="naive"):
    """
    Runs the backend simulation. 
    
    """
    bck = PseudoBackend(IR, PEs=PEs, placement=placement)
    instrs = [pe.instructions for pe in bck.processors]
    
    runner = Runtime(instrs)
    runner.run()

    print("Running with", PEs, "PEs took", runner.cycle, "cycles")
    print(runner.memory)
    
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--pe', type=int)
    parser.add_argument('--pl', type=str)
    
    args = parser.parse_args()
    
    txtfile = args.path
    PEs = args.pe
    placement = args.pl
    
    if not os.path.exists('./output'):
        os.makedirs('./output')


    # txtfile = './codeex.txt'
    with open(txtfile) as f:
        lines = f.readlines()

    # Initialize Graph
    G = DAG()
    # Initialize instruction dict, format {"instr_name": Instruction object}
    instructions = {}

    for line in lines:

        # Use Regex to extract all variables and operands (expression below assumes arithmetic operation)
        regex = re.compile(r'(\w+)\s*=\s*(\w+)\s*([+\-*/%×÷])\s*(\w+)')
        out = regex.match(line)

        # If there is no match, its a load/store operation
        if not out:
            # Re-write regex to match a load/store operation
            regex = re.compile(r'(\w+)\s*=\s*(\w+)')
            out = regex.match(line).groups()
            # Create an instruction object and add to dict
            instr = Instruction(out)
            instructions[instr.name] = instr

        # If match, its an arithmetic operation
        else:
            out = out.groups()
            # Create an instruction object and add to dict
            instr = Instruction(out)
            instructions[instr.name] = instr

    # Generate Graph, where each node is an Instruction object. This will serve as our IR.
    for name, instr in instructions.items():
        G.add_node(instr)
        for dep in instr.variables:
            G.add_edge(instructions[dep], instr)

    s = Source(G.get_dot(), filename="./output/dag.gv", format="png")
    s.save()
    s.render()

    simulator(G, PEs, placement)




