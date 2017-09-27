
from .main import symbol_associator, impl_bank
from ..inference.infer import FunctionAImpl
from ..front import parse_function


######################
# Projection helpers #
######################


def proj(psym):
    projectors = impl_bank['project']
    projs = projectors.setdefault(psym, {})

    @symbol_associator('proj')
    def pimpl(sym, name, fn):
        fsym, fenv = parse_function(fn)
        for s, lbda in fenv.bindings.items():
            impl_bank['abstract'][s] = \
                FunctionAImpl(lbda, impl_bank['abstract'])
        projs[impl_bank['abstract'][sym]] = \
            FunctionAImpl(fenv[fsym], impl_bank['abstract'])
        return fn

    return pimpl
