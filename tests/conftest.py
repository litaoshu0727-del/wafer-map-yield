# Import PyTorch (if installed) before scikit-learn. On macOS, a pip-installed
# torch and a conda-installed scikit-learn each ship their own OpenMP runtime,
# and loading scikit-learn's first makes torch crash on its first CPU kernel.
try:
    import torch  # noqa: F401
except ImportError:
    pass
