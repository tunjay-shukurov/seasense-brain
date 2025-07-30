## Project To-Do List

---

### ✅ Completed

- [x] Upgrade `README.md`  
    → Enhanced documentation explaining project goals, setup, and usage.

- [x] Refactor All Scripts  
    → Reorganized all code to be cleaner, modular, and efficient.

- [x] Write `mkdir.py` for Directory Setup  
    → Script for generating all required folders (`logs/`, `models/`, etc.).

---

### 🛠️ In Progress / Remaining Tasks

- [ ] **Make All Scripts Functional**  
    → Refactor into pure functions to improve modularity and testability.

- [ ] **Upgrade `data_harvester.py`**  
    → Refactor logic for better structure, fault-tolerance, and maintainability.

    #### 🔧 `data_harvester.py` Will Be Split Into Submodules:

    - [ ] `logging_utils.py`  
        Functions:
        - `log(msg)` – Append timestamped messages  
        - `log_event_block(text)` – Log formatted event blocks  
        - `log_run_header(label, path)` – Log a summary header for each run  
        - `write_log_details(path, event_list)` – Summarize downloaded events  

    - [ ] `download_utils.py`  
        Functions:
        - `fetch_events(...)` – Query events via IRIS FDSN  
        - `get_station_triplets(...)` – Filter stations by required channels  
        - `download_event_streams(...)` – Download waveform data and write SAC files  
        - `get_channel_list(type)` – Map `"HH?" → [HHZ, HH1, HH2]`, etc.  

    - [ ] `directory_manager.py`  
        Functions:
        - `ensure_base_dirs()` – Create `logs/`, `data/`, `models/` folders if missing  
        - `prepare_save_dir(...)` – Determine target directory for saving data  

    - [ ] `labeling.py`  
        Functions:
        - `assign_label(magnitude)` – Assign `"earthquake"` or `"noise"` based on magnitude  
        - Future: Add ML-based dynamic labeling logic here  

---

- [ ] **Refactor `train_ann.py`**  
    → Modularize the training logic, data loading, and model saving.

- [ ] **Improve Remaining Scripts**  
    → Refactor `predictor.py`, `simulate_data.py`, etc. to meet the same standards.

- [ ] **Create `main.py` as Central Pipeline Controller**  
    → Integrate all components: harvesting → preprocessing → training → evaluation.

- [ ] **Simplify Complex Scripts**  
    → Split large scripts into smaller modules under `helper/`, `core/`, or `io/` folders.

- [ ] **Improve Logging System**  
    → Build a richer and more readable logging format. Possibly with `comment.py` or `logger.py`.

- [ ] **Combine ANN and CNN Models**  
    → Implement model selector (e.g. ensemble, voting, or confidence threshold) and integrate into pipeline.

---

📝 **Last Updated:** July 30, 2025
