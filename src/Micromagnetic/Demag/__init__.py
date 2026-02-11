from .Demag_Lindholm import DemagField_Lindholm
from .Demag_FMM import DemagField_FMM
from .Demag_Bempp import DemagField_bempp
from .Lindholm_kernel import lindholm_weights_precomp, solid_angle_with_atan2
# Compatibility: DemagField targets the default method
DemagField = DemagField_Lindholm

_DEMAG_REGISTRY = {
    "lindholm": DemagField_Lindholm,
    "fmm": DemagField_FMM,
    "bempp": DemagField_bempp,
}

def make_demag_field(method, mesh, V, V1, Ms, **kwargs):
    key = (method or "lindholm").strip().lower()
    if key not in _DEMAG_REGISTRY:
        raise ValueError(
            f"demag_method='{method}' Not supported. Options: {list(_DEMAG_REGISTRY)}"
        )
    return _DEMAG_REGISTRY[key](mesh, V, V1, Ms, **kwargs)
