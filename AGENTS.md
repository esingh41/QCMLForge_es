# AGENTS.md - Guidelines for QCMLForge

## Build/Lint/Test Commands

### Testing
- **Run all tests**: `python -m pytest tests/`
- **Run single test file**: `python -m pytest tests/test_ap2.py`
- **Run specific test**: `python -m pytest tests/test_ap2.py::test_function_name -v`
- **Run with coverage**: `python -m pytest tests/ --cov=src/apnet_pt`

### Training
- **Train AtomModel**: `python train_models.py --train_am --am_model_path ./models/am_example.pt --n_epochs 5`
- **Train APNet2**: `python train_models.py --train_ap2 --ap_model_path ./models/ap2_example.pt --n_epochs 5`

### Environment Setup
- **Create conda env**: `conda env create -f environment.yml`
- **Install package**: `pip install -e .`

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports second
- Local imports last
- Use relative imports within the package
- Group imports with blank lines between groups

```python
import numpy as np
import torch
import qcelemental as qcel

from . import constants
from .util import helper_function
```

### Naming Conventions
- **Functions/Methods**: `snake_case` (e.g., `load_dimer_dataset`)
- **Classes**: `PascalCase` (e.g., `APNet2Model`, `AtomMPNN`)
- **Constants**: `ALL_CAPS` (e.g., `AU2ANG`, `ELEM_TO_Z`)
- **Variables**: `snake_case` (e.g., `model_path`, `batch_size`)

### Documentation
- Use triple-quoted docstrings for all public functions/classes
- Follow numpy docstring format
- Include parameter types and descriptions
- Document return values and exceptions

```python
def process_molecule(mol_data):
    """Convert molecule data to ML-ready format.

    Parameters
    ----------
    mol_data : dict
        Raw molecule data from QCArchive

    Returns
    -------
    np.ndarray
        Processed molecular features
    """
```

### Error Handling
- Use try/except blocks for expected errors
- Return `None` for invalid inputs rather than raising exceptions
- Log warnings for non-critical issues
- Use specific exception types when possible

```python
try:
    result = process_data(data)
except ValueError as e:
    logger.warning(f"Invalid data: {e}")
    return None
```

### Code Structure
- Keep functions under 50 lines when possible
- Use descriptive variable names
- Add comments for complex logic
- Follow PEP 8 formatting (4 spaces indentation)
- Use type hints for function parameters and return values

### File Organization
- Place utility functions in `util.py`
- Keep constants in `constants.py`
- Group related classes in modules (e.g., `AtomModels/`, `AtomPairwiseModels/`)
- Use `__init__.py` files to expose public APIs

### Testing
- Write unit tests for all public functions
- Use descriptive test names (e.g., `test_process_valid_molecule`)
- Include edge cases and error conditions
- Mock external dependencies when possible
- Place test data in `tests/test_data/` directory

### Performance
- Use vectorized operations with numpy/torch
- Avoid unnecessary loops
- Use GPU acceleration when available
- Profile code for bottlenecks before optimizing