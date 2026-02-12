# StateManager Migration Strategy

## Overview

This document outlines the strategy for migrating from scattered `st.session_state` references to centralized StateManager.

## Current State

**Files using session_state:**
- `ntpn/ntpn_utils.py` - 74+ references
- `pages/import_and_load_page.py`
- `pages/train_model_page.py`
- `pages/ntpn_visualisations_page.py`

**StateManager Status:**
- ✅ Created: `ntpn/state_manager.py`
- ✅ Tested: 22 tests, 98.15% coverage
- ✅ All tests passing: 50/50

## Migration Approach

### Phase 1: Initialization & Compatibility Layer

**Goal:** Initialize StateManager without breaking existing code

**Steps:**
1. Initialize StateManager at app startup
2. Create backward compatibility helpers
3. Verify app still works

**Files to modify:**
- `NTPN_APP.py` - Add StateManager initialization
- `ntpn/state_manager.py` - Add compatibility layer

### Phase 2: Migrate Core Utilities

**Goal:** Migrate `ntpn_utils.py` functions to use StateManager

**Strategy:** One function at a time, with tests after each

**Functions to migrate (in order):**
1. `initialise_session()` - State initialization
2. `load_demo_session()` - Demo data loading
3. `session_select()` - Session selection
4. `samples_transform()` - Data transformation
5. `create_trajectories()` - Trajectory creation
6. `create_model()` - Model creation
7. `compile_model()` - Model compilation
8. `train_model()` - Training wrapper
9. `train_for_streamlit()` - Custom training loop
10. `generate_critical_sets()` - Critical set generation
11. `generate_upper_bounds()` - Upper bounds generation
12. `visualise_critical_sets()` - Visualization

### Phase 3: Migrate Pages

**Goal:** Update UI pages to use StateManager

**Order (by complexity):**
1. `pages/import_and_load_page.py` - Simple data loading
2. `pages/train_model_page.py` - Model training
3. `pages/ntpn_visualisations_page.py` - Visualizations
4. `pages/mapper_page.py` - Mapper topology (if uses session_state)
5. `pages/vrtda_page.py` - VR topology (if uses session_state)

### Phase 4: Remove Compatibility Layer

**Goal:** Clean up backward compatibility code

**Steps:**
1. Remove legacy access methods from StateManager
2. Remove any temporary wrapper functions
3. Update documentation

## Migration Pattern

### Before (Old Pattern):
```python
def some_function():
    # Direct session_state access
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None

    data = st.session_state.dataset
    st.session_state.processed_data = process(data)
```

### After (New Pattern):
```python
def some_function(state: StateManager):
    # Type-safe StateManager access
    if not state.data.is_loaded():
        st.warning("Please load data first")
        return

    data = state.data.dataset
    state.data.select_samples = process(data)
```

## Session State Mapping

### Data State
| Old session_state key | New StateManager path |
|----------------------|----------------------|
| `dataset` | `state.data.dataset` |
| `labels` | `state.data.labels` |
| `dataset_name` | `state.data.dataset_name` |
| `select_samples` | `state.data.select_samples` |
| `select_indices` | `state.data.select_indices` |
| `select_labels` | `state.data.select_labels` |
| `sub_samples` | `state.data.sub_samples` |
| `sub_indices` | `state.data.sub_indices` |
| `sub_labels` | `state.data.sub_labels` |

### Model State
| Old session_state key | New StateManager path |
|----------------------|----------------------|
| `ntpn_model` | `state.model.ntpn_model` |
| `train_tensors` | `state.model.train_tensors` |
| `test_tensors` | `state.model.test_tensors` |
| `batch_size` | `state.model.batch_size` |
| `learning_rate` | `state.model.learning_rate` |
| `train_loss` | `state.model.train_loss` |
| `train_accuracy` | `state.model.train_accuracy` |
| `test_loss` | `state.model.test_loss` |
| `test_accuracy` | `state.model.test_accuracy` |

### Visualization State
| Old session_state key | New StateManager path |
|----------------------|----------------------|
| `cs_lists` | `state.viz.cs_lists` |
| `cs_pca` | `state.viz.cs_pca` |
| `cs_umap` | `state.viz.cs_umap` |
| `cs_cca_aligned` | `state.viz.cs_cca_aligned` |
| `upper_lists` | `state.viz.upper_lists` |
| `num_critical_samples` | `state.viz.num_critical_samples` |
| `pca_dimensions` | `state.viz.pca_dimensions` |
| `umap_n_neighbors` | `state.viz.umap_n_neighbors` |
| `umap_min_dist` | `state.viz.umap_min_dist` |

### UI State
| Old session_state key | New StateManager path |
|----------------------|----------------------|
| `current_page` | `state.ui.current_page` |
| `show_advanced_options` | `state.ui.show_advanced_options` |
| `debug_mode` | `state.ui.debug_mode` |
| `operation_in_progress` | `state.ui.operation_in_progress` |
| `progress_message` | `state.ui.progress_message` |

## Testing Strategy

After each migration step:
1. Run full test suite: `pytest tests/ -v`
2. Check coverage: `pytest tests/ --cov=ntpn --cov-report=html`
3. Manual smoke test:
   - Load demo data
   - Select sessions
   - Train model (1 epoch)
   - Generate critical sets
   - Visualize with PCA

## Success Criteria

- ✅ All tests pass
- ✅ No direct `st.session_state` access outside StateManager
- ✅ App functionality unchanged
- ✅ Type safety enforced by MyPy
- ✅ Code is more maintainable

## Rollback Plan

If migration causes issues:
1. Git revert to last working commit
2. Review failed step
3. Fix issue
4. Retry migration

## Current Progress

- ✅ StateManager created and tested
- ⏳ Phase 1: Initialization - NEXT
- ⏳ Phase 2: Migrate utilities
- ⏳ Phase 3: Migrate pages
- ⏳ Phase 4: Cleanup

**Last Updated:** 2026-02-11
