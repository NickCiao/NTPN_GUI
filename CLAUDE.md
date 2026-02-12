# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NTPN (Neural Trajectory Point Net) is a spatial transformer-based neural network for extracting and analyzing the geometry of neural representations from electrophysiology recordings. This repository contains a Streamlit GUI application for training models, visualizing critical sets/upper bounds, and performing topological data analysis on neural trajectories.

## Running the Application

### Using Docker (Recommended)

```bash
# Using Docker Compose
docker-compose up --build

# Or using Docker directly
docker build -t ntpn-gui .
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ntpn-gui
```

The app will be available at http://localhost:8501

See `DOCKER.md` for detailed Docker instructions.

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit GUI
streamlit run NTPN_APP.py

# Run with specific port
streamlit run NTPN_APP.py --server.port 8501
```

## Architecture

### Package Structure

```
ntpn/                          # Core package (no Streamlit dependency except ntpn_utils)
├── point_net.py               # PointNet model architecture
├── data_processing.py         # Data I/O, sampling, preprocessing, transforms, splitting
├── analysis.py                # UMAP, PCA, CS processing, CCA alignment
├── plotting.py                # All matplotlib visualization functions
├── data_service.py            # Service layer: data pipeline operations
├── model_service.py           # Service layer: model creation, compilation, training
├── visualization_service.py   # Service layer: critical set generation, plotting
├── ntpn_utils.py              # Streamlit-specific utilities (4 functions only)
├── state_manager.py           # Centralized type-safe state management
├── data_loaders.py            # Safe data loading (NPZ format with fallback)
├── ntpn_constants.py          # All configuration constants
├── logging_config.py          # Structured logging configuration
├── point_net_utils.py         # Backward-compat facade (re-exports from split modules)
└── __init__.py

pages/                         # Streamlit page modules
├── ntpn_landing_page.py       # Landing page with demo data initialization
├── import_and_load_page.py    # Data import, preprocessing, trajectory creation
├── train_model_page.py        # Model definition, compilation, training
├── ntpn_visualisations_page.py # Critical set generation and plotting
├── mapper_page.py             # Mapper topology visualization (stub — not yet implemented)
└── vrtda_page.py              # Vietoris-Rips persistent homology (stub — not yet implemented)

tests/
├── conftest.py                # Shared fixtures
├── fixtures/sample_data.py    # Test data generators
├── unit/                      # Unit tests for each module
├── integration/               # End-to-end pipeline tests
└── regression/                # Bug fix and backward-compat tests
```

### Core Components

**`NTPN_APP.py`** - Main entry point
- Multi-page Streamlit application using `st.navigation()`
- Initializes `StateManager` and structured logging at startup

**Service layer** (`data_service.py`, `model_service.py`, `visualization_service.py`)
- All business logic lives here, completely Streamlit-free
- Pages import directly from service modules (no facade indirection)
- Each service accepts an optional `StateManager` parameter

**`ntpn_utils.py`** - Streamlit-specific utilities (4 functions only)
- `initialise_session()` — loads demo data into StateManager
- `train_for_streamlit()` — custom training loop with Streamlit progress bars
- `train_model()` — dispatches between Streamlit and headless training
- `draw_image()` — renders images in Streamlit

**`state_manager.py`** - Centralized state management
- `DataState`, `ModelState`, `VisualizationState`, `UIState` dataclasses
- `StateManager` class with type-safe property access
- Singleton via `get_state_manager()`
- Backward-compatible `sync_to_legacy()` for `st.session_state`

**`point_net.py`** - PointNet architecture
- `point_net()`: Main model with two T-Net spatial transformers
- `tnet()`: Spatial transformer network
- `OrthogonalRegularizer`: Constrains transforms to near-orthogonal
- Ablation variants: `point_net_no_transform()`, `point_net_no_pool()`, `point_net_no_pool_no_transform()`
- `point_net_segment()`: Segmentation variant for point-wise predictions
- Critical/upper set extraction: `generate_critical()`, `generate_upper()`

**Split utility modules** (from the former monolithic `point_net_utils.py`):
- `data_processing.py` (~440 lines) — Data I/O, sampling, preprocessing, transforms, splitting
- `analysis.py` (~240 lines) — UMAP, PCA, CS processing, CCA alignment
- `plotting.py` (~360 lines) — All matplotlib visualization functions
- `point_net_utils.py` — Thin backward-compat facade re-exporting all 46 functions

### Data Flow Pipeline

1. **Data Loading** (`import_and_load_page.py` -> `data_service`)
   - Load binned spike counts (neurons x time bins) and labels
   - Safe NPZ format via `data_loaders.py` (with legacy .p fallback for user data)
   - Demo data: `data/demo_data/raw_stbins.npz` and `context_labels.npz` (NPZ only — legacy .p files removed)

2. **Data Preprocessing** (`import_and_load_page.py` -> `data_service`)
   - Session selection, noise removal, power/standard transforms
   - Sliding window trajectory creation
   - Neuron subsampling, train/test split into TensorFlow datasets

3. **Model Training** (`train_model_page.py` -> `model_service` + `ntpn_utils`)
   - Create and compile PointNet model
   - Train via Streamlit progress loop or headless Keras `fit()`
   - Save to `models/` directory as `.keras` files

4. **Visualization** (`ntpn_visualisations_page.py` -> `visualization_service`)
   - Critical set extraction and upper bounds generation
   - PCA or UMAP dimensionality reduction
   - CCA alignment across classes

5. **Topological Analysis** (`mapper_page.py`, `vrtda_page.py`) — *not yet implemented*
   - Mapper graph-based topology visualization (planned)
   - Vietoris-Rips complex persistent homology (planned)
   - TDA dependencies (kmapper, ripser, persim, scikit-tda) are in requirements.txt for future use

### Key Design Patterns

**Import pattern**: Pages import directly from service modules:
```python
from ntpn.data_service import session_select, create_trajectories
from ntpn.model_service import create_model, compile_model
from ntpn.visualization_service import generate_critical_sets
```

**State management**: All state flows through `StateManager`:
```python
from ntpn.state_manager import get_state_manager
state = get_state_manager()
state.data.dataset       # type-safe access
state.model.ntpn_model
```

**Model Architecture Invariances**
- **Permutation invariant**: Order of neurons doesn't matter (max pooling)
- **Transformation invariant**: Robust to rotation, translation, scaling (T-Net transformers)
- **Identity agnostic**: Works across sessions/subjects without neuron sorting

**Layer Naming Convention**
- Critical set extraction relies on specific layer names
- When modifying model architecture, update layer names in `visualization_service.generate_critical_sets()`

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ntpn --cov-report=html

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/regression/ -v

# Skip slow tests (UMAP-based)
pytest tests/ -v -m "not slow"
```

**Current stats:** 331 tests, 89.3% coverage

## Linting and Formatting

```bash
# Lint (auto-fix where possible)
ruff check --fix ntpn/ pages/ tests/

# Format
ruff format ntpn/ pages/ tests/
```

Configuration in `pyproject.toml`: Python 3.11+, 120 char lines, single quotes.

## Dependencies

See `requirements.txt` for full list. Key libraries:
- `streamlit` — GUI framework
- `tensorflow` / `keras` — Neural network
- `numpy`, `pandas`, `scipy` — Numerical computing
- `scikit-learn` — PCA, CCA, preprocessing, splitting
- `umap-learn` — UMAP dimensionality reduction
- `kmapper`, `ripser`, `persim`, `scikit-tda` — Topological data analysis (for future use)
- `matplotlib`, `Pillow` — Visualization
- `pytest`, `ruff`, `mypy` — Development tools

## Development Notes

### Adding New Pages

1. Create `pages/new_page.py` with a main function called at module level
2. Register in `NTPN_APP.py`: `new_page = st.Page('pages/new_page.py', title='Page Title')`
3. Add to navigation list in `st.navigation([...])`

### Model Architecture Changes

- Update layer width/depth in `point_net.py`
- Verify layer names for critical set extraction (`model.summary()`)
- Update `layer_name` parameter in `visualization_service.generate_critical_sets()`
- Test with ablation variants

### Data Format Requirements

Input data must be:
- Binned spike counts: List of numpy arrays, shape (neurons, time_bins)
- Labels: List of numpy arrays with class labels per time bin
- Preferred format: NPZ files via `data_loaders.py`
