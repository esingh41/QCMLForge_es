# AGENTS.md - Guidelines for QCMLForge

## Build/Lint/Test Commands
- **Test all**: `python -m pytest tests/`
- **Test single file**: `python -m pytest tests/test_ap2.py`
- **Test specific**: `python -m pytest tests/test_ap2.py::test_function_name -v`
- **Test with coverage**: `python -m pytest tests/ --cov=src/apnet_pt`
- **Train AtomModel**: `python train_models.py --train_am --am_model_path ./models/am_example.pt --n_epochs 5`
- **Train APNet2**: `python train_models.py --train_ap2 --ap_model_path ./models/ap2_example.pt --n_epochs 5`
- **Install**: `pip install -e .` or `conda env create -f environment.yml`

## Code Style Guidelines
- **Imports**: stdlib → third-party → local; use relative imports within package
- **Naming**: `snake_case` functions/vars, `PascalCase` classes, `ALL_CAPS` constants
- **Docs**: NumPy-style docstrings with parameter/return types
- **Error handling**: Return `None` for invalid inputs, use specific exceptions
- **Structure**: PEP 8, 4-space indent, type hints, functions <50 lines
- **Organization**: Utils in `util.py`, constants in `constants.py`, tests in `tests/`
- **Performance**: Vectorized numpy/torch ops, avoid loops, use GPU when available