# AGENTS.md

## Cursor Cloud specific instructions

This repo is a single Streamlit app (NTPN — Neural Trajectory Point Net). See `README.md` and `CLAUDE.md` for architecture and the canonical lint/test/run commands.

### Environment
- Python dependencies are installed into a local virtualenv at `.venv` (Python 3.12). The startup update script recreates/refreshes it from `requirements.txt`, so it is available at the start of every session.
- Prefix commands with the venv, e.g. `.venv/bin/streamlit`, `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/python`.
- System packages `python3.12-venv` and `build-essential` are required to create the venv / build some wheels; they are already present in the environment snapshot (do not add them to the update script).

### Running the app
- Start the GUI: `.venv/bin/streamlit run NTPN_APP.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true`. Health check: `curl -s http://localhost:8501/_stcore/health` returns `ok`.
- The demo dataset (`data/demo_data/*.npz`) is loaded lazily by `initialise_session()`, which runs **only** on the landing page (`NTPN Application`). Always open the landing page first. If you jump straight to `Import Models and Data`, `state.data.dataset` is `None` and the page throws `TypeError: 'NoneType' object is not iterable` until the landing page has run once.

### Known behavior gotchas (pre-existing code, do not "fix" as part of setup)
- On the `Import Models and Data` → `Pre-processing` tab, select **two or more** sessions. Selecting a single session with "Remove Noise Category" checked hits a pre-existing bug in `ntpn/data_processing.remove_noise_cat`/`precut_noise` (`IndexError: list index out of range`) because the single-selection branch passes the whole list instead of one session's array.
- The demo subset is tiny, so training can print `Training Acc: 0.0000` and Keras may warn "input ran out of data"; this is expected with the demo data, not an environment problem.

### Lint / test
- Lint: `.venv/bin/ruff check ntpn/ pages/ tests/`. The pinned floor is `ruff>=0.4.0`, but the installed ruff is newer and reports ~29 pre-existing style findings (e.g. `B905`, `F841`); these are not caused by setup.
- Tests: `.venv/bin/pytest tests/ -q` (331 tests pass, ~90% coverage; writes an `htmlcov/` report). Some UMAP tests are marked `slow`; skip with `-m "not slow"`.

### Full pipeline (headless smoke test)
The end-to-end pipeline (load demo → session_select(2+ sessions) → transform → create_trajectories → create_train_test → create_model → compile_model → train_model_headless) runs headlessly via the functions in `ntpn/data_service.py` and `ntpn/model_service.py`; `StateManager` stores state in `st.session_state`, so a headless script must set `streamlit.session_state = {}` before use.
