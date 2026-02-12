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

### Core Components

**`NTPN_APP.py`** - Main entry point
- Multi-page Streamlit application using `st.navigation()`
- Manages page routing to different workflow stages
- Imports from `ntpn/` package for core functionality

**`ntpn/` package** - Core neural network implementation
- **`point_net.py`**: PointNet architecture with spatial transformers
  - `point_net()`: Main model with two T-Net transformers
  - `tnet()`: Spatial transformer network that aligns inputs/features
  - `OrthogonalRegularizer`: Custom regularizer constraining transforms to near-orthogonal
  - Ablation variants: `point_net_no_transform()`, `point_net_no_pool()`, `point_net_no_pool_no_transform()`
  - `point_net_segment()`: Segmentation variant for point-wise predictions
  - Visualization functions: `predict_critical()`, `predict_upper()`, `generate_critical()`, `generate_upper()`
- **`point_net_utils.py`**: Data processing and visualization utilities
  - Data loading from pickle files
  - Sliding window projection for trajectory creation
  - Neuron subsampling and train/test splitting
  - PCA, UMAP, and CCA alignment for critical sets
  - Plotting functions for trajectories and critical sets
- **`ntpn_utils.py`**: Streamlit-specific application utilities
  - Session state management
  - Data pipeline orchestration (session selection, transformation, trajectory creation)
  - Custom training loop for Streamlit progress display
  - Critical set generation and visualization workflows
- **`ntpn_constants.py`**: Configuration constants
  - Demo data file paths
  - Default parameters

**`pages/` directory** - Streamlit page modules
- Each page implements a specific workflow stage
- Pages are automatically loaded by `st.Page()` in NTPN_APP.py
- Standard pattern: each page defines a main function that's called at module level

### Data Flow Pipeline

The NTPN workflow follows this sequence:

1. **Data Loading** (`import_and_load_page.py`)
   - Load binned spike counts (neurons × time bins) and labels from pickle files
   - Demo data: `data/demo_data/raw_stbins.p` and `context_labels.p`
   - Session state stores: `dataset`, `labels`

2. **Data Preprocessing** (`train_model_page.py`)
   - Session selection: Choose which experimental sessions to include
   - Noise removal: Optional filtering via `remove_noise_cat()`
   - Transform: Apply power transform or standard scaling
   - Trajectory creation: Sliding window over time to create fixed-length trajectories
   - Neuron subsampling: Randomly sample subset of neurons for each trajectory
   - Train/test split: Generate TensorFlow datasets

3. **Model Training** (`train_model_page.py`)
   - Create PointNet model with specified architecture (layer width, trajectory length)
   - Compile with optimizer and loss function
   - Train using either standard Keras `fit()` or custom Streamlit loop for progress display
   - Save trained models to `models/` directory as `.keras` files

4. **Visualization** (`ntpn_visualisations_page.py`)
   - **Critical sets**: Extract points in trajectories that maximally activate each feature (contribute to max pooling)
   - **Upper bounds**: Find boundary of feature space for each class
   - Dimensionality reduction: PCA or UMAP for 3D visualization
   - CCA alignment: Align critical sets across classes

5. **Topological Analysis**
   - **Mapper** (`mapper_page.py`): Graph-based topology visualization using filtered point clouds
   - **VR-Complexes** (`vrtda_page.py`): Vietoris-Rips complex analysis for persistent homology

### Key Design Patterns

**Session State Management**
- Streamlit's `st.session_state` stores all data, models, and intermediate results
- `ntpn_utils.initialise_session()` sets up initial state with demo data
- Key state variables: `dataset`, `labels`, `ntpn_model`, `sub_samples`, `train_tensors`, `cs_lists`

**Model Architecture Invariances**
- **Permutation invariant**: Order of neurons doesn't matter (max pooling)
- **Transformation invariant**: Robust to rotation, translation, scaling (T-Net transformers)
- **Identity agnostic**: Works across sessions/subjects without neuron sorting

**Layer Naming Convention**
- Critical set extraction relies on specific layer names (e.g., `'activation_14'` for the last activation before max pooling)
- When modifying model architecture, update layer names in `generate_critical_sets()`

## Dependencies

Core libraries (inferred from imports):
- `streamlit` - GUI framework
- `tensorflow` / `keras` - Neural network implementation
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Preprocessing (PCA, CCA, StandardScaler, train/test split)
- `umap-learn` - UMAP dimensionality reduction
- `giotto-tda` - Time series (SlidingWindow) and topological data analysis
- `matplotlib` - Visualization
- `scipy`, `scikit-image`, `Pillow` - Additional utilities

## Development Notes

### Adding New Pages

To add a new Streamlit page:
1. Create `pages/new_page.py` with a main function called at module level
2. Register in `NTPN_APP.py`: `new_page = st.Page('pages/new_page.py', title='Page Title')`
3. Add to navigation list: `pg = st.navigation([..., new_page])`

### Model Architecture Changes

If modifying the PointNet architecture:
- Update layer width/depth in `point_net.py`
- Verify layer names for critical set extraction (check with `model.summary()`)
- Update `layer_name` parameter in `ntpn_utils.generate_critical_sets()` if needed
- Test with ablation variants to understand impact

### Data Format Requirements

Input data must be:
- Binned spike counts: List of numpy arrays, shape (neurons, time_bins)
- Labels: List of numpy arrays with class labels per time bin
- Stored as pickle files with specific dictionary structure (see `load_data_pickle()`)

### Training Display

Two training modes:
- `view=True`: Custom training loop with Streamlit progress bars (`train_for_streamlit()`)
- `view=False`: Standard Keras `model.fit()` for faster training without UI updates
