
from .ast import Symbol, Value
from .util import Props
from typing import Dict


def bsym(name):
    return Symbol(name, namespace='builtin')


def gsym(name):
    return Symbol(name, namespace='global')


builtins_dict: Dict[str, Symbol] = dict(
    # Builtins that correspond to operator applications,
    # or are inserted by the compiler. They are not accessible
    # in the user's global namespace
    add = bsym('add'),
    subtract = bsym('subtract'),
    multiply = bsym('multiply'),
    divide = bsym('divide'),
    power = bsym('power'),
    dot = bsym('dot'),
    bitwise_or = bsym('bitwise_or'),
    bitwise_and = bsym('bitwise_and'),
    bitwise_xor = bsym('bitwise_xor'),
    unary_add = bsym('unary_add'),
    unary_subtract = bsym('unary_subtract'),
    bitwise_not = bsym('bitwise_not'),
    negate = bsym('negate'),
    less = bsym('less'),
    greater = bsym('greater'),
    less_equal = bsym('less_equal'),
    greater_equal = bsym('greater_equal'),
    equal = bsym('equal'),
    index = bsym('index'),
    getattr = bsym('getattr'),
    setslice = bsym('setslice'),
    lazy_if = bsym('lazy_if'),
    half_lazy_if = bsym('half_lazy_if'),
    identity = bsym('identity'),

    # Myia's global variables
    myia_builtins = gsym('myia_builtins'),
    len = gsym('len'),
    range = gsym('range'),
    enumerate = gsym('enumerate'),
    map = gsym('map'),
    filter = gsym('filter'),
    switch = gsym('switch'),
    first = gsym('first'),
    second = gsym('second'),
)


builtins = Props(builtins_dict)


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


function_map = {
    range: builtins.range,
}


# Not yet used [[[BEGIN

# _maps = {
#     'builtin': True,
#     'numpy': False
# }


# def _add_numpy_map():
#     # Note: we will only run this if numpy has already been
#     # imported by the user
#     import numpy as _
#     numpy_map = {
#         _.add: builtins.add,
#         _.arange: builtins.range,
#         _.divide: builtins.divide,
#         _.dot: builtins.dot,
#         _.multiply: builtins.multiply,
#         _.subtract: builtins.subtract
#     }
#     numpy_map[_] = Value(_)

#     global function_map
#     function_map = {**function_map, **numpy_map}


# def _update_function_map():
#     # This populates function_map with <function> => <Symbol> mappings,
#     # e.g. numpy.add => Symbol("+")
#     # However, we don't want to add mappings for a package if that package
#     # has not been imported by the user, so we first check if it is present
#     # in sys.modules, then we call the corresponding _add function
#     # defined above.
#     for package, added in _maps.items():
#         if not added and sys.modules[package]:
#             # TODO: error handling
#             globals()['_add_{}_map'.format(package)]()
#             _maps[package] = True

# END]]]


def get_operator(node):
    try:
        return operator_map[node.__class__.__name__]
    except KeyError:
        raise NotImplementedError("Unknown operator: {}".format(node))
