
import networkx as nx
from ..symbols import builtins, inst_builtin
from ..stx import Transformer, \
    Symbol, ValueNode, ApplyNode, LambdaNode, LetNode, \
    ClosureNode, TupleNode, MyiaSyntaxError, TMP, \
    is_global, is_builtin, GenSym, create_lambda
from ..util import Singleton
from types import FunctionType
from ..lib import Record, ZERO
from ..parse import parse_function
from numpy import ndarray
from ..transform import a_normal
from copy import copy
from buche import buche
from ..inference.types import var, unify, isvar
from collections import defaultdict


ogen = GenSym('global::optimized')


###############
# Edge labels #
###############


# x ==FN=> y      : y is the function used to compute x
# x ==CL=> y      : y is the function for closure x
# x ==IN(i)=> y   : y is the ith argument to the func that computes x
# x ==ELEM(i)=> y : x[i] is y
# x ==GET(i)=> y  : x is y[i]


class FN(Singleton):
    pass


class CL(Singleton):
    pass


class COPY(Singleton):
    pass


FN = FN()  # type: ignore
CL = CL()  # type: ignore
COPY = COPY()  # type: ignore


class IN:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f'IN({self.index})'

    def __hash__(self):
        return hash(IN) ^ self.index

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.index == other.index


class ELEM:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f'.{self.index}='

    def __hash__(self):
        return hash(ELEM) ^ self.index

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.index == other.index


class GET:
    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f'.{self.index}'

    def __hash__(self):
        return hash(GET) ^ self.index

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.index == other.index


#####################
# Computation types #
#####################


class Operation:
    def install(self, graph, node):
        graph.clear_out(node)
        node.operation = self


class ApplyOperation(Operation):
    def __init__(self, fn, args):
        self.fn = fn
        self.args = tuple(args)

    def install(self, graph, node):
        super().install(graph, node)
        graph.add_edge(node, self.fn, FN)
        for i, arg in enumerate(self.args):
            graph.add_edge(node, arg, IN(i))


# class ClosureOperation(Operation):
#     def __init__(self, fn, args):
#         self.fn = fn
#         self.args = tuple(args)

#     def install(self, graph, node):
#         super().install(graph, node)
#         graph.add_edge(node, self.fn, CL)
#         for i, arg in enumerate(self.args):
#             graph.add_edge(node, arg, IN(i))


def ClosureOperation(fn, args):
    return ApplyOperation(IRValue(builtins.partial), (fn,) + tuple(args))


# class TupleOperation(Operation):
#     def __init__(self, values):
#         self.values = tuple(values)

#     def install(self, graph, node):
#         super().install(graph, node)
#         for i, value in enumerate(self.values):
#             graph.add_edge(node, value, ELEM(i))


def TupleOperation(values):
    return ApplyOperation(IRValue(builtins.mktuple), values)


# class GetOperation(Operation):
#     def __init__(self, arg, index):
#         self.arg = arg
#         self.index = index

#     def install(self, graph, node):
#         super().install(graph, node)
#         graph.add_edge(node, self.arg, GET(self.index))


def GetOperation(arg, index):
    return ApplyOperation(IRValue(builtins.index), (arg, IRValue(index)))


# class CopyOperation(Operation):
#     def __init__(self, arg):
#         self.arg = arg

#     def install(self, graph, node):
#         super().install(graph, node)
#         graph.add_edge(node, self.arg, COPY)


def CopyOperation(arg):
    return ApplyOperation(IRValue(builtins.identity), (arg,))


#########################
# Graph class and nodes #
#########################


class IRGraph(nx.MultiDiGraph):

    def clear_out(self, node):
        if not self.has_node(node):
            return
        prev = list(self.edges(node, keys=True))
        if prev:
            self.remove_edges_from(prev)

    def link(self, dest, source, role):
        assert not isinstance(dest, Symbol)
        assert not isinstance(source, Symbol)
        self.add_edge(dest, source, role)

    def sync(self, node):
        if not self.has_node(node):
            self.add_node(node)
        e = self.edges(node, keys=True)
        fn = None
        cl = None
        cp = None
        inputs = {}
        elems = {}
        idx = None
        for n, node2, key in e:
            if key is FN:
                fn = node2
            elif key is CL:
                cl = node2
            elif key is COPY:
                cp = node2
            elif isinstance(key, IN):
                inputs[key.index] = node2
            elif isinstance(key, ELEM):
                elems[key.index] = node2
            elif isinstance(key, GET):
                idx = (node2, key.index)
            else:
                pass
        if len([x for x in [fn, cl, elems, idx, cp] if x]) > 1:
            raise Exception('Same node cannot be produced in multiple ways.')
        if fn:
            args = tuple(inputs[i] for i in range(len(inputs)))
            op = ApplyOperation(fn, args)
        elif cl:
            args = tuple(inputs[i] for i in range(len(inputs)))
            op = ClosureOperation(cl, args)
        elif elems:
            op = TupleOperation(tuple(elems[i] for i in range(len(elems))))
        elif idx:
            op = GetOperation(*idx)
        elif cp:
            op = CopyOperation(cp)
        else:
            op = None
        node.operation = op

    def productor(self, node):
        self.sync(node)
        return node.operation

    def toposort(self):
        return nx.topological_sort(self)


class IRNode:
    def __init__(self, tag):
        self.tag = tag
        self.values = None
        self.operation = None

    def to_ast(self):
        return self.tag

    def rename(self, gen):
        if isinstance(self.tag, ValueNode):
            return self
        if isinstance(self.tag, Symbol) and is_global(self.tag):
            return self
        n = copy(self)
        n.tag = gen.dup(self.tag)
        n.operation = None
        return n

    @property
    def value(self):
        v, = self.values
        return v

    def has_value(self):
        return self.values and len(self.values) == 1

    def __str__(self):
        return str(f'{self.__class__.__name__}({self.tag})')

    __repr__ = __str__

    def __hrepr__(self, H, hrepr):
        return hrepr(self.tag)


class IRInput(IRNode):
    pass


class IRComputation(IRNode):
    pass


class IRValue(IRNode):
    def __init__(self, value):
        super().__init__(value)
        self.values = [value]

    def to_ast(self):
        v = self.value
        if isinstance(v, Symbol):
            return v
        else:
            return ValueNode(v)


X = var('X')
Y = var('Y')
Z = var('Z')
V = var('V', lambda x: x.has_value())
V1 = var('V1', lambda x: x.has_value())
V2 = var('V2', lambda x: x.has_value())
L = var('L', lambda x: isinstance(x, IRLambda))


class PatternOpt:
    def __init__(self, pattern, handler):
        self.pattern = pattern
        self.handler = handler

    def _match(self, lbda, node, pattern, U):
        if isinstance(pattern, tuple):
            prod = lbda.graph.productor(node)
            if isinstance(prod, ApplyOperation):
                sexp = (prod.fn,) + prod.args
                if ... in pattern:
                    idx = pattern.index(...)
                    ntail = len(pattern) - idx - 1
                    pattern = pattern[:idx] + pattern[idx + 1:]
                    idx_last = len(sexp) - ntail
                    mid = list(sexp[idx - 1:idx_last])
                    sexp = sexp[:idx - 1] + (mid,) + sexp[idx_last:]
                if len(sexp) != len(pattern):
                    return False
                for p, e in zip(pattern, sexp):
                    U = self._match(lbda, e, p, U)
                    if U is False:
                        return False
                return U
            else:
                return False
        elif isinstance(node, list):
            if isvar(pattern):
                for x in node:
                    if not unify(pattern, x, U):
                        return False
                return {**U, var(pattern.token): node}
            else:
                return False
        elif isvar(pattern):
            return unify(pattern, node, U)
        elif node.has_value():
            return unify(pattern, node.value, U)
        else:
            return unify(pattern, node, U)

    def match(self, lbda, node):
        return self._match(lbda, node, self.pattern, {})

    def __call__(self, lbda, node):
        m = self.match(lbda, node)
        if m:
            kwargs = {k.token: v for k, v in m.items()}
            repl = self.handler(lbda, node, **kwargs)
            if repl is True or repl is False:
                return repl
            elif isinstance(repl, Operation):
                repl.install(lbda.graph, node)
                return True
            else:
                lbda.replace(node, repl)
                return True
        return False


all_patterns = {}


def pattern_opt(*pattern):
    def wrap(handler):
        opt = PatternOpt(pattern, handler)
        all_patterns[handler.__name__] = opt
        return opt
    return wrap


@pattern_opt(builtins.add, (builtins.mktuple, X, ...),
                           (builtins.mktuple, Y, ...))
def add_tuples(lbda, node, X, Y):
    adds = []
    for x, y in zip(X, Y):
        tmp = IRComputation(ogen('T'))
        op = ApplyOperation(IRValue(builtins.add), (x, y))
        op.install(lbda.graph, tmp)
        adds.append(tmp)
    return TupleOperation(adds)


@pattern_opt(builtins.multiply, 1.0, X)
def multiply_by_one(lbda, node, X):
    return X


@pattern_opt(builtins.index, (builtins.mktuple, X, ...), V)
def take_index(lbda, node, X, V):
    return X[V.value]


@pattern_opt(builtins.J, (builtins.switch, X, Y, Z))
def J_switch(lbda, node, X, Y, Z):
    tmps = []
    for v in (X, Y, Z):
        t = IRComputation(ogen(v.tag, 'T'))
        tmps.append(t)
        ApplyOperation(IRValue(builtins.J), (v,)).install(lbda.graph, t)
    return ApplyOperation(IRValue(builtins.switch), tmps)


@pattern_opt(builtins.Jinv, (builtins.J, X))
def Jinv_J_cancel(lbda, node, X):
    return X


@pattern_opt(builtins.J, (builtins.Jinv, X))
def J_Jinv_cancel(lbda, node, X):
    return X


@pattern_opt(L, X, ...)
def inline(lbda, node, L, X):
    f = L
    args = X

    g = lbda.graph

    f2 = f.dup()
    g.add_edges_from(f2.graph.edges(keys=True))

    for input, arg in zip(f2.inputs, args):
        CopyOperation(arg).install(g, input)
    CopyOperation(f2.output).install(g, node)

    return True


@pattern_opt((builtins.partial, X, Y, ...), Z, ...)
def call_partial(lbda, node, X, Y, Z):
    return ApplyOperation(X, Y + Z)


@pattern_opt(V1, V2, ...)
def eval_constant(lbda, node, V1, V2):
    if V1.value == builtins.partial:
        return False
    f = V1
    args = V2
    eenv = lbda.universe.eenv
    fn = eenv.evaluate(f.value)
    res = fn(*[eenv.import_value(arg.value) for arg in args])
    return lbda.value_to_node(res)


@pattern_opt(builtins.identity, X)
def drop_copy(lbda, node, X):
    return X


@pattern_opt(L, V, ...)
def eval_constant2(lbda, node, V1, V2):
    f = L
    args = V
    eenv = lbda.universe.eenv
    fn = eenv.evaluate(f.lbda)
    res = fn(*[eenv.import_value(arg.value) for arg in args])
    return lbda.value_to_node(res)


# @pattern_opt(builtins.closure_args, X)
# def take_closure_args(lbda, node, X):
#     prod = lbda.graph.productor(X)
#     if isinstance(prod, ClosureOperation):
#         return TupleOperation(prod.args)
#     else:
#         return False


class IRLambda(IRNode):
    def __init__(self, universe, tag, gen, inputs, output, lbda):
        super().__init__(tag)
        self.universe = universe
        self.gen = gen
        self.inputs = inputs
        self.output = output
        self.lbda = lbda
        self.graph = IRGraph()
        self.values = [lbda]

    def dup(self):
        g = self.graph
        corresp = {n: n.rename(self.gen) for n in g.nodes}
        for i in self.inputs:
            corresp.setdefault(i, i.rename(self.gen))
        lbda = copy(self)
        g2 = nx.relabel_nodes(g, corresp, copy=True)
        lbda.graph = IRGraph()
        lbda.graph.add_edges_from(g2.edges(keys=True))
        # for node in lbda.graph.nodes:
        #     lbda.graph.sync(node)
        lbda.inputs = [corresp[i] for i in self.inputs]
        lbda.output = corresp[self.output]
        return lbda

    def prune(self):
        to_remove = set(self.graph.nodes)
        to_remove -= set(nx.dfs_preorder_nodes(self.graph, self.output))
        if to_remove:
            self.graph.remove_nodes_from(to_remove)
            return True
        return False

    def merge(self, n1, n2):
        if n1 is n2:
            return True
        if type(n1) != type(n2):
            return False
        if isinstance(n1, IRLambda):
            return False
        if self._is_constant(n1):
            if n1.value == n2.value:
                self.replace(n1, n2)
                return True
            return False
        else:
            edges1 = {k: n for _, n, k in self.graph.edges(n1, keys=True)}
            edges2 = {k: n for _, n, k in self.graph.edges(n2, keys=True)}
            if edges1.keys() != edges2.keys():
                return False
            for k, n3 in edges1.items():
                if not self.merge(n3, edges2[k]):
                    return False
            self.replace(n1, n2)
            return True

    def cse(self):
        node_to_hash = {}
        hash_to_node = defaultdict(list)
        for n in reversed(list(nx.topological_sort(self.graph))):
            if isinstance(n, (IRLambda, IRInput)):
                h = hash(n)
            elif self._is_constant(n):
                h = hash(n.value) ^ hash(type(n.value))
            else:
                h = 0
                for _, n2, k in self.graph.edges(n, keys=True):
                    h ^= node_to_hash[n2] ^ hash(k)
            node_to_hash[n] = h
            hash_to_node[h].append(n)
        for h, (node0, *nodes) in hash_to_node.items():
            for n in nodes:
                self.merge(n, node0)

    def replace(self, old, new):
        g = self.graph
        for n, _, k in list(g.in_edges(old, keys=True)):
            g.remove_edge(n, old, k)
            g.add_edge(n, new, k)
        if old is self.output:
            self.output = new

    def _is_constant(self, node):
        return isinstance(node, IRLambda) or \
            (node.values and len(node.values) == 1)

    def value_to_node(self, v):
        from ..interpret import Function
        from ..lib import Closure

        if isinstance(v, Function):
            an = a_normal(v.ast)
            an.primal = v.ast.primal
            return self.universe.acquire(an)
        if isinstance(v, Closure):
            cl = self.value_to_node(v.fn)
            n = IRComputation(ogen('<clos>'))
            self.graph.link(n, cl, CL)
            for i, arg in enumerate(v.args):
                a = self.value_to_node(arg)
                self.graph.link(n, a, IN(i))
            return n
        elif isinstance(v, (int, float, str)):
            return IRValue(v)
        else:
            raise Exception(f'Do not recognize value: {v}')

    def simplify(self):
        changes = False
        for n in list(self.graph.nodes):
            for opt in all_patterns.values():
                changes |= opt(self, n)
        return changes


class GlobalResolver(dict):
    # TODO: merge this with EvaluationEnv

    def __init__(self, object_map, pool):
        self.object_map = object_map
        self.pool = pool
        self.accepted_types = (bool, int, float,
                               ndarray, list, tuple, Record, str)

    def import_value(self, v):
        try:
            x = self.object_map[v]
        except (TypeError, KeyError):
            pass
        else:
            return self.import_value(x)

        if isinstance(v, Symbol) and is_builtin(v):
            return v

        # try:
        #     x = self.primitives[v]
        # except (TypeError, KeyError):
        #     pass
        # else:
        #     return x

        if isinstance(v, self.accepted_types) or v is ZERO or v is None:
            return v
        elif isinstance(v, (type, FunctionType)):
            # Note: Python's FunctionType, i.e. actual Python functions
            try:
                lbda = parse_function(v)
            except (TypeError, OSError):
                raise ValueError(f'Myia cannot interpret value: {v}')
            return self.import_value(lbda)
        elif isinstance(v, LambdaNode):
            return v
        elif isinstance(v, Symbol):
            raise ValueError(f'Myia cannot resolve {v} '
                             f'from namespace {v.namespace}')
        else:
            raise ValueError(f'Myia cannot interpret value: {v}')

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            if isinstance(item, Symbol) and is_builtin(item):
                self[item] = item
            else:
                self[item] = self.import_value(self.pool[item])
            return self[item]


class Universe:
    def __init__(self, resolver, eenv):
        self.resolver = resolver
        self.eenv = eenv
        self.anf_lbdas = {}
        self.dependencies = nx.DiGraph()

    def acquire(self, anf_lbda):
        if anf_lbda in self.anf_lbdas:
            return self.anf_lbdas[anf_lbda]
        irlbda = self._graph_transform(anf_lbda)
        # # NOTE: _graph_transform sets the key in self.anf_lbdas
        # self.anf_lbdas[anf_lbda] = irlbda
        return irlbda

    def aggressively_optimize(self, anf_lbda):
        lbda = self.anf_lbdas[anf_lbda]
        changes = True
        while changes:
            # Run until equilibrium (all the optimizations below
            # are normalizing)
            opers = [lbda.simplify(),
                     lbda.cse()]
            changes = any(opers)
            lbda.prune()

    def _graph_transform(self, lbda):
        let = lbda.body
        assert isinstance(let, LetNode)

        assoc = {}

        def wrap(x, cls=IRComputation):
            if not isinstance(x, (Symbol, ValueNode)):
                raise Exception('Should be a Symbol or ValueNode.', x)
            values = None

            if x in assoc:
                return assoc[x]

            if isinstance(x, ValueNode):
                values = {x.value}
            elif is_global(x):
                if is_builtin(x):
                    values = {x}
                else:
                    imp = self.resolver[x]
                    if isinstance(imp, LambdaNode):
                        imp2 = a_normal(imp)
                        imp2.primal = imp.primal
                        irlbda = self.acquire(imp2)
                        assoc[x] = irlbda
                        return irlbda
                    elif isinstance(imp, Symbol) and is_builtin(imp):
                        values = {imp}
                    else:
                        raise Exception('Invalid thing to import.')

            if isinstance(x, Symbol):
                n = assoc.setdefault(x, cls(x))
            else:
                n = cls(x)
            n.values = values
            return n

        inputs = [wrap(sym, IRInput) for sym in lbda.args]
        output = wrap(let.body, IRComputation)
        irl = IRLambda(self, ogen(lbda.ref, '!'), lbda.gen,
                       inputs, output, lbda)

        self.anf_lbdas[lbda] = irl
        self.dependencies.add_node(irl)

        g = irl.graph

        def assign(k, v, rel=None):
            if isinstance(k, TupleNode):
                tmp = lbda.gen(TMP)
                assign(tmp, v, rel)
                for i, kk in enumerate(k.values):
                    assign(kk, tmp, GET(i))
                return

            wk = wrap(k)
            if rel is not None:
                g.link(wk, wrap(v), rel)
                return

            if isinstance(v, ApplyNode):
                args = [wrap(a) for a in v.args]
                ApplyOperation(wrap(v.fn), args).install(g, wk)
            elif isinstance(v, (Symbol, ValueNode)):
                CopyOperation(wrap(v)).install(g, wk)
            elif isinstance(v, ClosureNode):
                args = [wrap(a) for a in v.args]
                ClosureOperation(wrap(v.fn), args).install(g, wk)
            elif isinstance(v, TupleNode):
                args = [wrap(a) for a in v.values]
                TupleOperation(args).install(g, wk)
            else:
                raise MyiaSyntaxError('Illegal ANF clause.', node=v)

        for k, v in let.bindings:
            assign(k, v)

        return irl

    @property
    def lbdas(self):
        return list(self.dependencies.nodes)

    def export(self, anf_lbda):
        irlbda = self.anf_lbdas[anf_lbda]
        return GraphToANF().transform(irlbda)

    def export_all(self):
        return {lbda: self.export(lbda) for lbda in self.anf_lbdas.keys()}


class GraphToANF:
    def __init__(self):
        pass

    def transform(self, irl):
        g = irl.graph
        nodes = g.toposort()
        bindings = []

        for node in nodes:
            assert not isinstance(node, Symbol)

            if isinstance(node, IRLambda):
                pass

            else:
                prod = g.productor(node)
                if isinstance(prod, ApplyOperation):
                    args = [a.to_ast() for a in prod.args]
                    fn = prod.fn.to_ast()
                    if fn == builtins.identity:
                        val, = args
                    elif fn == builtins.mktuple:
                        val = TupleNode(args)
                    elif fn == builtins.partial:
                        fn, *args = args
                        val = ClosureNode(fn, args)
                    else:
                        val = ApplyNode(fn, *args)
                else:
                    continue
                bindings.append((node.tag, val))
        bindings.reverse()

        l = create_lambda(irl.tag,
                          [i.tag for i in irl.inputs],
                          LetNode(bindings, irl.output.tag),
                          irl.gen)
        l.primal = irl.lbda.primal
        return l
