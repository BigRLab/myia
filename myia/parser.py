"""Parse a Python AST into the Myia graph-based ANF IR.

Graph construction proceeds very similarly to the way that FIRM constructs its
SSA graph [1]_. The correspondence between functional representations and SSA
is discussed in detail in [2]_. The main takeaway is that basic blocks
correspond to functions, and jumping from one block to another is done by a
tail call. Phi nodes become formal parameters of the function. The inputs to a
phi node become actual parameters (arguments) passed at the call sites in the
predecessor blocks.

Note that variable names in this module closely follow the ones used in [1]_ in
order to make it easy to compare the two algorithms.

The `process_>` functions generally take two arguments: The current
block, and the AST node to convert to ANF in the context of this block. The
`process_statement` functions will return the current block at the end of
processing (which might have changed), whereas `process_expression` functions
will return the ANF node corresponding to the value of the expression that was
processed.

.. [1] Braun, M., Buchwald, S. and Zwinkau, A., 2011. Firm-A graph-based
   intermediate representation. KIT, Fakultät für Informatik.
.. [2] Appel, A.W., 1998. SSA is functional programming. ACM SIGPLAN Notices,
   33(4), pp.17-20.

"""
import ast
import inspect
import textwrap
from typing import Dict, List

from myia.anf_ir import ANFNode, Parameter, Apply, Graph, Constant
from myia.primops import If, Add, Return

RETURNS = []

CONSTANTS = {
    ast.Add: Constant(Add()),
}

IF_OP = Constant(If())
RETURN_OP = Constant(Return())


class Block:
    """A basic block.

    A basic block is used while parsing the Python code to represent a segment
    of code (e.g. a function body, loop body, a branch). During parsing it
    keeps track of a variety of information needed to resolve variable names.

    Attributes:
        variables: A mapping from variable names to the nodes representing the
            bound value at this point of parsing. If a variable name is not in
            this mapping, it needs to be resolved in the predecessor blocks.
        phi_nodes: A mapping from parameter nodes corresponding to phi nodes to
            variable names. Once all the predecessor blocks (calling functions)
            are known, we can resolve these variable names in the predecessor
            blocks to find out what the arguments at the call site are.
        jumps: A mapping from successor blocks to the function calls that
            correspond to these jumps. This is information that was not used in
            the FIRM algorithm; it is necessary here because it is not possible
            to distinguish regular function calls from the tail calls used for
            control flow.
        matured: Whether all the predecessors of this block have been
            constructed yet. If a block is not mature and a variable cannot be
            resolved, we have to construct a phi node (i.e. add a parameter to
            this function). Once the block is mature, we will resolve the
            variable in the parent blocks and use them as arguments.
        preds: The predecessor blocks of this block.
        graph: The ANF function graph corresponding to this basic block.

    """

    def __init__(self) -> None:
        """Construct a basic block.

        Constructing a basic block also constructs a corresponding function,
        and a constant that can be used to call this function.

        """
        self.matured: bool = False
        self.variables: Dict[str, ANFNode] = {}
        self.preds: List[Block] = []
        self.phi_nodes: Dict[Parameter, str] = {}
        self.jumps: Dict[Block, Apply] = {}
        self.graph: Graph = Graph()
        CONSTANTS[self.graph] = Constant(self.graph)

    def set_phi_arguments(self, phi: Parameter) -> None:
        """Resolve the arguments to a phi node.

        Args:
            phi: The `Parameter` node which is functioning as a phi node. The
                arguments corresponding to this parameter will be read from
                predecessor blocks (functions).

        """
        varnum = self.phi_nodes[phi]
        for pred in self.preds:
            arg = pred.read(varnum)
            jump = pred.jumps[self]
            jump.inputs.append(arg)
        # TODO remove_unnecessary_phi(phi)

    def mature(self) -> None:
        """Mature this block.

        A block is matured once all of its predecessor blocks have been
        processed. This triggers the resolution of phi nodes.

        """
        # Use the function parameters to ensure proper ordering.
        for phi in self.graph.parameters:
            if phi in self.phi_nodes:
                self.set_phi_arguments(phi)
        self.matured = True

    def read(self, varnum: str) -> ANFNode:
        """Read a variable.

        If this name has defined given in one of the previous statements, it
        will be trivially resolved. It is possible that the variable was
        defined in a previous block (e.g. outside of the loop body or the
        branch). In this case, it will be resolved only if all predecessor
        blocks are available. If they are not, we will assume that this
        variable is given as a function argument (which plays the role of a phi
        node).

        Args:
            varnum: The name of the variable to read.

        """
        if varnum in self.variables:
            return self.variables[varnum]
        if self.matured and len(self.preds) == 1:
            return self.preds[0].read(varnum)
        phi = Parameter(self.graph)
        self.graph.parameters.append(phi)
        self.phi_nodes[phi] = varnum
        self.write(varnum, phi)
        if self.matured:
            self.set_phi_arguments(phi)
        return phi

    def write(self, varnum: str, node: ANFNode) -> None:
        """Write a variable.

        When assignment is used to bound a value to a name, we store this
        mapping in the block to be used by subsequent statements.

        Args:
            varnum: The name of the variable to store.
            node: The node representing this value.

        """
        self.variables[varnum] = node

    def jump(self, target: 'Block') -> Apply:
        """Jumping from one block to the next becomes a tail call.

        This method will generate the tail call by calling the graph
        corresponding to the target block using an `Apply` node, and returning
        its value with a `Return` node. It will update the predecessor blocks
        of the target appropriately.

        Args:
            target: The block to jump to from this statement.

        """
        jump = Apply([CONSTANTS[target.graph]], self.graph)
        self.jumps[target] = jump
        target.preds.append(self)
        return_ = Apply([RETURN_OP, jump], self.graph)
        self.graph.return_ = return_
        return return_

    def cond(self, cond: ANFNode, true: 'Block', false: 'Block') -> Apply:
        """Perform a conditional jump.

        This method will generate the call to the if expression and return its
        value. The predecessor blocks of the branches will be updated
        appropriately.

        Args:
            cond: The node representing the condition (true or false).
            true: The block to jump to if the condition is true.
            false: The block to jump to if the condition is false.

        """
        true.preds.append(self)
        false.preds.append(self)
        inputs = [IF_OP, cond, CONSTANTS[true.graph], CONSTANTS[false.graph]]
        if_ = Apply(inputs, self.graph)
        return_ = Apply([RETURN_OP, if_], self.graph)
        self.graph.return_ = return_
        return return_


def process_function(block: Block, node: ast.FunctionDef) -> Block:
    """Process a function definition.

    Args:
        block: Predecessor block (optional). If given, this is a nested
            function definition.
        node: The function definition.

    """
    function_block = Block()
    if block:
        function_block.preds.append(block)
    function_block.mature()
    function_block.graph.debug.name = node.name
    function_block.graph.debug.ast = node
    for arg in node.args.args:
        anf_node = Parameter(function_block.graph)
        anf_node.debug.name = arg.arg
        anf_node.debug.ast = arg
        function_block.graph.parameters.append(anf_node)
        function_block.write(arg.arg, anf_node)
    return process_statements(function_block, node.body)


def process_return(block: Block, node: ast.Return) -> Block:
    """Process a return statement."""
    return_ = Apply([RETURN_OP, process_expression(block, node.value)],
                    block.graph)
    block.graph.return_ = return_
    return_.debug.ast = node
    RETURNS.append(return_)
    return block


def process_assign(block: Block, node: ast.Assign) -> Block:
    """Process an assignment."""
    anf_node = process_expression(block, node.value)
    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        anf_node.debug.name = node.targets[0].id
        block.write(node.targets[0].id, anf_node)
    else:
        raise NotImplementedError(node.targets)
    return block


def process_expression(block: Block, node: ast.expr) -> ANFNode:
    """Process an expression."""
    expr: ANFNode
    if isinstance(node, ast.BinOp):
        inputs_: List[ANFNode] = [CONSTANTS[type(node.op)],
                                  process_expression(block, node.left),
                                  process_expression(block, node.right)]
        expr = Apply(inputs_, block.graph)
        expr.debug.ast = node
    elif isinstance(node, ast.Name):
        expr = block.read(node.id)
    elif isinstance(node, ast.Num):
        if node.n not in CONSTANTS:
            CONSTANTS[node.n] = Constant(node.n)
        expr = CONSTANTS[node.n]
    return expr


def process_statements(block: Block, nodes: List[ast.stmt]) -> Block:
    """Process a sequence of statements."""
    for node in nodes:
        block = process_statement(block, node)
    return block


def process_statement(block: Block, node: ast.stmt) -> Block:
    """Process a single statement."""
    if isinstance(node, ast.Assign):
        return process_assign(block, node)
    elif isinstance(node, ast.FunctionDef):
        return process_function(block, node)
    elif isinstance(node, ast.Return):
        return process_return(block, node)
    elif isinstance(node, ast.If):
        return process_if(block, node)
    elif isinstance(node, ast.While):
        return process_while(block, node)
    else:
        raise NotImplementedError(node)


def process_if(block: Block, node: ast.If) -> Block:
    """Process a conditional statement.

    A conditional statement generates 3 functions: The true branch, the false
    branch, and the continuation.

    """
    true_block = Block()
    false_block = Block()
    after_block = Block()
    cond = process_expression(block, node.test)
    true_block.mature()
    false_block.mature()

    # Process the first branch
    true_end = process_statements(true_block, node.body)
    true_end.jump(after_block)

    # And the second
    false_end = process_statements(false_block, node.orelse)
    false_end.jump(after_block)

    block.cond(cond, true_block, false_block)
    after_block.mature()
    return after_block


def process_while(block: Block, node: ast.While) -> Block:
    """Process a while loop.

    A while loop will generate 3 functions: The test, the body, and the
    continuation.

    """
    header_block = Block()
    body_block = Block()
    after_block = Block()

    block.jump(header_block)

    cond = process_expression(header_block, node.test)
    body_block.mature()
    header_block.cond(cond, body_block, after_block)
    after_body = process_statements(body_block, node.body)
    after_body.jump(header_block)
    header_block.mature()
    after_block.mature()
    return after_block


def parse(func):
    """Parse a function into ANF."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    process_statement(None, tree.body[0])
