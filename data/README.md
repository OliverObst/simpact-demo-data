# SIMPaCT demo data – data directory

This directory contains raw, processed, and analysis-generated files used in the SIMPaCT mutual-information-based virtual sensor prediction experiments.

## Files

### Raw Data
- **`sopa.csv`**  
  Raw volumetric water content (VWC) readings from 13 soil moisture sensors in Bicentennial Park, Sydney, Australia.  
  Columns:
    - `index` — Row index from original extract
    - `stream` — Sensor identifier and measurement type
    - `value` — VWC measurement (%)
    - `datetime` — UTC timestamp (ISO format)

### Processed Data
- **`sopivot+idx.csv`**  
  Pivoted dataset with sensors as columns, indexed by date and hour.
  Preprocessing:
    - Selected `.processed` sensor streams only
    - Converted to hourly averages
    - Simplified sensor IDs to `SENSxxxx`
    - Created a `(day, hour)` index and a `datetime` column

### Generated Analysis Files
These are outputs from the mutual information (MI) computation pipeline:
- **`mi_bundle.npz`** — Bundled numpy arrays for MI analysis
- **`mi_full.csv`** — Full mutual information matrix
- **`mi_meta.json`** — Metadata for sensors and analysis settings
- **`mi_norm.csv`** — Normalised MI values
- **`mi_pos.csv`** — Sensor position data
- **`mi_pvals.csv`** — P-values for MI values
- **`mi_top1_edges.csv`** — Top MI edges for each node
- **`mi_top1_graph.gpickle`** — Graph object of top MI edges

## Time Span
- Start: 2022-11-15 00:00 UTC  
- End: 2023-01-17 15:00 UTC  
- Frequency: Hourly (processed) / Irregular (raw)

## Sensor IDs
SENS0008, SENS0010, SENS0012, SENS0017, SENS0018, SENS0019, SENS0020, SENS0021, SENS0022, SENS0023, SENS0027, SENS0028, SENS0030.

## License
- **Raw and processed data**: CC BY 4.0 (see `LICENSE-DATA.md`)  
- **Generated analysis files**: CC BY 4.0  
- **Code**: MIT License

If you use this data in academic work, please cite our paper.
