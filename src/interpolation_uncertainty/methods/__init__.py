"""
methods/__init__.py — estimator registry
=========================================

All uncertainty estimators (built-in and user-defined) are stored in
``_REGISTRY``.  Every key is the string a caller passes as ``method=``.

Built-in registration
---------------------
The spectral estimators are registered below using ``functools.partial`` so
that each ``method=`` variant is its own callable with the correct signature::

    (signal, resolution, line_spacing, **kwargs) -> (energy_2d, frequencies_1d)

Plugin registration
-------------------
Drop a ``.py`` file anywhere inside this ``methods/`` package directory.
Inside that file, decorate your estimator function with ``@register``::

    # interpolation_uncertainty/methods/my_plugin.py
    import numpy as np
    from interpolation_uncertainty.methods import register

    @register('my_method')
    def my_estimator(signal, resolution, line_spacing, **kwargs):
        ...
        return energy, frequencies   # (n_segments, n_freqs), (n_freqs,)

The function will be available as ``method='my_method'`` in
``process_line_pair`` and ``build_uncertainty_raster`` automatically — no
other files need to be edited.

Public API
----------
- ``register(name, *, overwrite=False)`` — decorator
- ``get_estimator(name)``                — look up by method string
- ``list_estimators()``                  — sorted list of registered names
- ``compute_uncertainty``                — backward-compat alias for the
                                          full ``spectral_estimator`` callable
"""
import functools
import importlib
import pkgutil
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable] = {}
_discovered: bool = False


def register(name: str, *, overwrite: bool = False) -> Callable:
    """Register a function as an uncertainty estimator under *name*.

    Parameters
    ----------
    name : str
        Key used to select this estimator via ``method=`` arguments.
    overwrite : bool
        Allow replacing an existing registration. Default ``False``.

    Returns
    -------
    decorator : callable
        The decorated function is returned unchanged; only the registry entry
        is added as a side-effect.

    Raises
    ------
    KeyError
        If *name* is already registered and *overwrite* is ``False``.

    Examples
    --------
    ::

        @register('my_method')
        def my_estimator(signal, resolution, line_spacing, **kwargs):
            energy = ...          # shape (n_segments, n_freqs)
            frequencies = ...     # shape (n_freqs,)
            return energy, frequencies
    """
    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY and not overwrite:
            existing = _REGISTRY[name]
            raise KeyError(
                f"Estimator '{name}' is already registered by "
                f"{existing.__module__}.{existing.__qualname__}. "
                f"Pass overwrite=True to replace it."
            )
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_estimator(name: str) -> Callable:
    """Return the estimator callable registered under *name*.

    Triggers auto-discovery of any unimported ``methods/*.py`` plugin files
    on the first call, so plugins dropped into the directory are found
    without any manual import.

    Parameters
    ----------
    name : str
        Estimator key, e.g. ``'psd'``, ``'amplitude'``, ``'my_method'``.

    Returns
    -------
    estimator : callable
        Signature: ``(signal, resolution, line_spacing, **kwargs)``
        Returns: ``(energy_2d, frequencies_1d)``.

    Raises
    ------
    KeyError
        If *name* is not found after discovery.
    """
    _autodiscover()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown estimator '{name}'. "
            f"Available estimators: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name]


def list_estimators() -> list[str]:
    """Return a sorted list of all registered estimator names."""
    _autodiscover()
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def _autodiscover() -> None:
    """Import every ``methods/*.py`` module so their ``@register`` calls run.

    Safe to call multiple times — discovery happens at most once per process.
    Errors in a plugin are caught and re-raised with the module name so a
    broken plugin does not silently hide itself.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    methods_dir = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(methods_dir)]):
        if module_name.startswith('_'):
            continue
        full_name = f"interpolation_uncertainty.methods.{module_name}"
        try:
            importlib.import_module(full_name)
        except Exception as exc:
            raise ImportError(
                f"Failed to import estimator plugin '{full_name}': {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Register built-in spectral estimators
# ---------------------------------------------------------------------------

from .spectral import spectral_estimator  # noqa: E402

_SPECTRAL_METHODS = ('amplitude', 'psd', 'psd_n', 'psd_lf', 'psd_df', 'spectrum')
for _m in _SPECTRAL_METHODS:
    _REGISTRY[_m] = functools.partial(spectral_estimator, method=_m)

# Backward-compatible alias used by the package root __init__.py and notebooks
# compute_uncertainty = spectral_estimator


# ---------------------------------------------------------------------------
# Register built-in statistical estimators
# ---------------------------------------------------------------------------

from .statistical import statistical_estimator  # noqa: E402

_STATISTICAL_METHODS = ('SAR', 'SER', 'SMR')
for _m in _STATISTICAL_METHODS:
    _REGISTRY[_m] = functools.partial(statistical_estimator, method=_m)

# Backward-compatible alias used by the package root __init__.py and notebooks
# compute_uncertainty = statistical_estimator`