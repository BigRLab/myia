

from .ast import Transformer, Symbol
from .interpret import global_env, impl
from .symbols import builtins, bsym


builtins.zero = bsym('zero')
builtins.merge = bsym('merge')
builtins.J = bsym('J')
builtins.Jinv = bsym('Jinv')


######################################
# Decorator for gradient definitions #
######################################


_grad_map = {}


def rgrad(sym):
    # Copy symbol to grad namespace
    rsym = Symbol(sym.label, namespace='grad:builtin')

    def decorator(fn):
        _grad_map[sym] = rsym
        global_env[rsym] = fn
        return fn
    return decorator


################################################
# Implementation of primitives needed for Grad #
################################################


@impl(builtins.zero)
def zero(x):
    if isinstance(x, (int, float)):
        return 0
    elif isinstance(x, tuple):
        return tuple(gzero(a) for a in x)
    elif isinstance(x, FunctionImpl):
        return ()
    elif isinstance(x, ClosureImpl):
        return tuple(zero(a) for a in x.args)
    else:
        raise TypeError('Cannot create a zero conformant with {x}')


@impl(builtins.merge)
def merge(x, y):
    assert type(x) is type(y)
    if isinstance(x, (int, float)):
        return x + y
    elif isinstance(x, tuple):
        assert len(x) == len(y)
        return tuple(merge(a, b) for a, b in zip(x, y))
    else:
        raise TypeError('Cannot merge values of type {type(x)}')


@impl(builtins.J)
def J(x):
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return tuple(J(a) for a in x)
    elif isinstance(x, FunctionImpl):
        return make_grad(x)
    elif isinstance(x, ClosureImpl):
        # ??
        return ClosureImpl(J(x.fn), J(x.args))
    else:
        raise TypeError('Invalid argument for J: {x}')


@impl(builtins.Jinv)
def Jinv(x):
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, tuple):
        return tuple(Jinv(a) for a in x)
    elif isinstance(x, FunctionImpl):
        return x.primal
    elif isinstance(x, ClosureImpl):
        return ClosureImpl(Jinv(x.fn), Jinv(x.args))
    else:
        raise TypeError('Invalid argument for Jinv: {x}')


###########################################
# Gradients of primitives needed for Grad #
###########################################


@rgrad(builtins.zero)
def gzero(d, x):
    ...


@rgrad(builtins.merge)
def gmerge(d, x, y):
    ...


@rgrad(builtins.J)
def gJ(d, x):
    ...


@rgrad(builtins.Jinv)
def gJinv(d, x):
    ...


######################################
# Gradients of arithmetic primitives #
######################################


@rgrad(builtins.add)
def gadd(dz, _, _):
    return (dz, dz)


@rgrad(builtins.subtract)
def gsubtract(dz, _, _):
    return (dz, -dz)


@rgrad(builtins.multiply)
def gmultiply(dz, x, y):
    return (dz * y, dz * x)


@rgrad(builtins.divide)
def gdivide(dz, x, y):
    return (dz / y, -dz * x / (y * y))


# Following the methodology in the following paper:
#   http://www.bcl.hamilton.ie/~barak/papers/toplas-reverse.pdf

class Grad(Transformer):
    # Notation:
    # x_up is the reverse (backprop-ready) version of x
    # x_bprop is a function that takes the sensitivity of x and
    #     returns the sensitivity of the inputs of the function
    #     that returns x
    # x_sen is the sensitivity of the gradient to changes in x,
    #     i.e. the quantity we are ultimately interested in

    def __init__(self):
        self.sensitivity_map = {}
        self.backpropagator_map = {}

    def phi(var, value):
        # phi (p. 26) transformation on let bindings, transforms
        # the forward phase.

        if isinstance(value, Symbol):
            # x = y ==> x_up = y_up
            return [(self.transform(var), self.transform(value))]

        elif isinstance(value, Apply):
            # x = f(y) ==> (x_up, x_bprop) = f_up(y_up)
            tmp = self.gensym('tmp')
            return [(tmp,
                     Apply(self.transform(value.fn),
                           *[self.transform(a) for a in value.args])),
                    (self.transform(var),
                     Apply(builtins.index, tmp, Value(0))),
                    (self.backpropagator_var(var),
                     Apply(builtins.index, tmp, Value(0)))]

        elif isinstance(value, Closure):
            # x = lambda y: ... ==> x_up = (lambda y: ...)_up
            # But in our system, we feed free variables explicitly
            # through Closure, and lambda has no freevars, so we do:
            # x = Closure(f, w, z) ==> x_up = Closure(f_up, w_up, z_up) (???)

            args = [self.transform(a) for a in value.args]
            return [(self.transform(var),
                     Closure(self.transform(value.fn), args))]

        else:
            raise Exception(f'phi is not defined on node type: {value}')

    def rho(var, value):
        # rho (p. 26) transformation on let bindings, represents the
        # corresponding operations to do in the backward phase

        if isinstance(value, Symbol):
            # x = y ==> y_sen += x_sen
            return self.accum([value], self.sensitivity_var(var))

        elif isinstance(value, Apply):
            # x = f(y) ==> (f_sen, y_sen) += x_bprop(x_sen)
            args = [value.fn, *value.args]
            increment = Apply(self.backpropagator_var(var),
                              self.sensitivity_var(var))
            return self.accum(args, increment)

        elif isinstance(value, Closure):
            # x = Closure(f, w, z) ==> (w_sen, z_sen) += x_sen
            return self.accum(value.args, var)

        else:
            raise Exception(f'rho is not defined on node type: {value}')

    def zero_init(var):
        return (self.new_sensitivity_var(var),
                Apply(builtins.zero, self.transform(var)))

    def accum(vars, value):
        if isinstance(vars, list):
            sens = list(map(self.sensitivity_var, vars))
            new_sens = list(map(self.new_sensitivity_var, vars))
            tmp = self.gensym('tmp')
            group = Tuple(sens)
            app = Apply(builtins.gadd, group, value)
            rval = [(tmp, app)]
            for i, new_sen in enumerate(new_sens):
                rval.append((new_sen, Apply(builtins.index, tmp, Value(i))))
            return rval
        else:
            sen = self.sensitivity_var(var)
            new_sen = self.new_sensitivity_var(var)
            app = Apply(builtins.gadd, sen, value)
            return [(new_sen, app)]

    def sensitivity_var(v):
        # Maps v to the v_sen variable i.e. the gradient of v
        return self.sensitivity_map[v]

    def new_sensitivity_var(v):
        # Create a new sensitivity variable for v. This is used to preserve
        # the single-assignment property: instead of v_sen = v_sen + x,
        # we do v_sen2 = v_sen + x. self.sensitivity_var maps to the latest
        # return value for this function.
        new_v = self.gensym(v.label)
        self.sensitivity_map[v] = new_v
        return new_v

    def backpropagator_var(v):
        # Maps v to the v_bprop variable i.e. the backpropagator for v
        return self.backpropagator_map.setdefault(v, self.gensym(v.label))

    # def transform_Value(self, node):
    #     return node

    # def transform_Apply(self, node):
    #     return node

    # def transform_Closure(self, node):
    #     return node

    # def transform_If(self, node):
    #     return node

    # def transform_Lambda(self, node):
    #     return node

    def transform_Let(self, node):
        forward = []
        backward = []
        for s, v in node.bindings:
            forward += self.phi(s, v)
            backward += self.rho(s, v)

        zeros = [self.zero_init(s) for s in the_arguments] \
            + [self.zero_init(s) for s, _ in node.bindings]

        out_sen = self.sensitivity_var(node.body)

        backp_args = Tuple(...)
        backp_ret = Tuple(self.sensitivity_var(arg) for arg in self.backp_args)
        backp = Lambda([*backp_args, out_sen], Let(backward, backp_ret))

        return Let(forward, Tuple(self.transform(node.body), backp))

    # def transform_Symbol(self, node):
    #     return node

    # def transform_Tuple(self, node):
    #     return node
