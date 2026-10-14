# Numerical inputs

The authors' archived MAIP inputs are packaged losslessly in three NPZ files.
Use `numpy.load(filename, allow_pickle=False)` to read them. Source-file SHA-256
values and distributed-file hashes are in `manifest.json`.

| File | Arrays |
|---|---|
| `circular.npz` | Original sensitivity `(104, 3228)`, positions `(3228, 4)`, grid-to-mesh and mesh-to-grid projection matrices |
| `thorax.npz` | Original sensitivity `(104, 3413)`, positions `(3413, 4)`, grid-to-mesh and mesh-to-grid projection matrices |
| `voltages.npz` | Eight named arrays, each `(104, 4)` float32, in original measurement/frequency order |

The model files contain `sensitivity` (float64, as read from the original TXT),
`positions` (integer position table), `eit2rect` and `rect2eit` (float32).
`eit2rect` has shape `(4096, n)` and `rect2eit` has shape `(n, 4096)`. Position
table columns 3 and 4 are one-based image row and column indices.
The original sensitivities are normalized by the reconstruction code, not during
packaging. Voltage files retain their float32 values as loaded by the original
inference code; no extra sign change, noise or normalization is applied.

| Case / key in `voltages.npz` | Source file | Geometry |
|---|---|---|
| `case1_sim` | `V_case1_sim.txt` | Circular |
| `case2_sim` | `V_case2_sim.txt` | Circular |
| `case3_sim` | `V_case3_sim.txt` | Circular |
| `test_sim` | `V_test_sim.txt` | Circular |
| `DBLung_sim` | `V_DBLung_sim.txt` | Thorax |
| `Lung_realworld` | `V_Lung_realworld.txt` | Thorax |
| `test_realworld` | `V_test_realworld.txt` | Circular |
| `zebrafish_realworld` | `V_zebrafish_realworld.txt` | Circular |

`case2_sim` is identical to `code/simulation_data/V_P6.txt`, the default archived
entry point. Case names and frequency-column order are retained. Physical
frequency values and conductivity ground truth were not included with these
source inputs. Real-world files contain numerical voltage measurements only.

All data use the root [MIT license](../LICENSE). Please cite the MAIP paper:
[10.1109/JIOT.2025.3624228](https://doi.org/10.1109/JIOT.2025.3624228).
