"""
Symbol bank for Myia. Each symbol represents a certain
functionality, but does not contain it. A symbol is a
glorified variable name.

The symbols live in two namespaces:

* ``builtin`` is the namespace for functions that are for
  internal use and are not meant to be referred to by name
  by the user.
* ``global`` is the namespace for global functions that the
  user can refer to by name.
"""


import ast
import sys
from .stx import Symbol, ValueNode, bsym, gsym
from .util import Props, SymbolsMeta
from typing import Dict


def is_global(sym):
    ns = sym.namespace
    return ns == 'builtin' or ns.startswith('global')


gsym = bsym
class builtins(metaclass=SymbolsMeta):
    add = gsym('add')
    subtract = bsym('subtract')
    multiply = bsym('multiply')
    divide = bsym('divide')
    power = bsym('power')
    dot = bsym('dot')
    bitwise_or = bsym('bitwise_or')
    bitwise_and = bsym('bitwise_and')
    bitwise_xor = bsym('bitwise_xor')
    unary_add = bsym('unary_add')
    unary_subtract = bsym('unary_subtract')
    bitwise_not = bsym('bitwise_not')
    negate = bsym('negate')
    less = bsym('less')
    greater = bsym('greater')
    less_equal = bsym('less_equal')
    greater_equal = bsym('greater_equal')
    equal = bsym('equal')
    index = bsym('index')
    getattr = bsym('getattr')
    setslice = bsym('setslice')
    identity = bsym('identity')
    Closure = gsym('Closure')
    closure_fn = gsym('closure_fn')
    closure_args = gsym('closure_args')
    mktuple = bsym('mktuple')

    # Grad-related builtins
    fill = gsym('fill')
    zeros_like = gsym('zeros_like')
    ones_like = gsym('ones_like')
    J = gsym('J')
    Jinv = gsym('Jinv')

    # Myia's global variables
    myia_builtins = gsym('myia_builtins')
    raise_exception = gsym('raise_exception')
    Exception = bsym('Exception')
    print = gsym('print')
    len = gsym('len')
    range = gsym('range')
    enumerate = gsym('enumerate')
    map = gsym('map')
    reduce = gsym('reduce')
    filter = gsym('filter')
    switch = gsym('switch')
    first = gsym('first')
    second = gsym('second')

    # For type system
    assert_true = gsym('assert_true')
    type = gsym('type')
    shape = gsym('shape')


# Maps the names of Python AST nodes to corresponding
# builtin operations.
operator_map: Dict[str, Symbol] = dict(
    Add = builtins.add,
    Sub = builtins.subtract,
    Mult = builtins.multiply,
    Div = builtins.divide,
    Pow = builtins.power,
    MatMult = builtins.dot,
    BitOr = builtins.bitwise_or,
    BitAnd = builtins.bitwise_and,
    BitXor = builtins.bitwise_xor,
    UAdd = builtins.unary_add,
    USub = builtins.unary_subtract,
    Invert = builtins.bitwise_not,
    Not = builtins.negate,
    Lt = builtins.less,
    Gt = builtins.greater,
    LtE = builtins.less_equal,
    GtE = builtins.greater_equal,
    Eq = builtins.equal
    # NotEq = builtins.
    # In = builtins.
    # NotIn = builtins.
    # Is = builtins.
    # IsNot = builtins.
)


object_map = {
    len: builtins.len,
    range: builtins.range,
    map: builtins.map,
    filter: builtins.filter,
    enumerate: builtins.enumerate,
    Exception: builtins.Exception,
    # Closure: builtins.Closure,
    # closure_args: builtins.closure_args,
    # closure_fn: builtins.closure_fn
}


_maps = {
    'builtin': True,
    'numpy': False
}


def _add_numpy_map():
    # Note: we will only run this if numpy has already been
    # imported by the user
    import numpy as _
    numpy_map = {
        _.add: builtins.add,
        _.arange: builtins.range,
        _.divide: builtins.divide,
        _.dot: builtins.dot,
        _.multiply: builtins.multiply,
        _.subtract: builtins.subtract
    }
    numpy_map[_] = ValueNode(_)

    global object_map
    object_map = {**object_map, **numpy_map}


def update_object_map():
    # This populates object_map with <object> => <Symbol/Value> mappings,
    # e.g. numpy.add => Symbol("+")
    # However, we don't want to add mappings for a package if that package
    # has not been imported by the user, so we first check if it is present
    # in sys.modules, then we call the corresponding _add function
    # defined above.
    for package, added in _maps.items():
        if not added and package in sys.modules:
            # TODO: error handling
            globals()['_add_{}_map'.format(package)]()
            _maps[package] = True


def get_operator(node: ast.AST) -> Symbol:
    """
    Given a Python AST node, return the corresponding Symbol.
    """
    try:
        return operator_map[node.__class__.__name__]
    except KeyError:
        raise NotImplementedError("Unknown operator: {}".format(node))
