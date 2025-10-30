# Vendor Network v2

## Scientific Overview
This project simulates complex market dynamics between customers and a landscape of local and digital vendors. It features:
- **Classic Model (classic_model.py):** Agent-based, explicit stochastic simulation.
- **Mean-Field Model (mean_field_model.py):** Physics-inspired, self-consistent Ising/Boltzmann model.

Three global controls map market phases:
- Temperature (`T`, choice noise, softmax)
- Social coupling (`J`) — collective field/influence strength
- Visibility (`K`) — information bandwidth/consideration set (number of vendors seen per choice)

Both engines share utility/math infrastructure for apples-to-apples physical analysis. Phase diagrams, critical points, and order transitions can be mapped in (T, J, K) space.

## Folder Layout
- `customer_generator.py` – Generate customer population datasets (correlated price/urgency sensitivity, loyalty, social influence)
- `vendor_generator.py` – Generate vendor population datasets (local/digital split, cost-based pricing, delays, ratings)
- `classic_model.py` – Agent-based simulation model (was market_model.py)
- `mean_field_model.py` – Mean-field/physics model (was vendor_choice_model.py)
- `utility.py` – Shared utility function/masking and softmax logic
- `run_sim.py` – Unified runner (loads agents, sets (T, J, K), outputs diagrams)
- `plot_phase.py` – For advanced analysis and plotting
- `data/` – Population and output CSVs/NPZ

## Setup
1. Clone/download this folder and enter it.
2. Install requirements (see below).

## Requirements
All major dependencies are listed in `requirements.txt` (create using pip freeze/conda list from a working environment).
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Usage
1. **Generate Agent Populations:**
   ```bash
   python customer_generator.py
   python vendor_generator.py
   ```
   (Outputs written to `data/`)
2. **Run Simulation:**
   ```bash
   python run_sim.py
   ```
   (Add argument or config to select model flavor/classic/meanfield as needed)
3. **Plot/Analyze:**
   ```bash
   python plot_phase.py
   ```

## Extending the Model
- See in-file comments and docstrings for tips on adding new agent attributes, upgrading physics, or experimenting with new plot types/order parameters.
- Contributions aiming for scientific publication are encouraged—modularize and document all physics/math improvements.

## Reproducibility
- Output file names and seed settings should be preserved for all production jobs to ensure results traceability.
- Use separate parameter/config yaml/json to keep runs reproducible (optional, implement as needed).

## Project Led By
- Principal: [Your Name]
- AI Assistant: ChatGPT + code audit tools
- For publication targets, cite relevant agent-based modeling/complex systems literature as needed.
