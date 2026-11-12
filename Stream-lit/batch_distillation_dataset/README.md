# Batch Distillation Dataset

## Overview

Real laboratory recordings of a mixture of **butan-1-ol, propan-2-ol, and water**. Batch distillation separates these liquids by heating the mixture and cooling the resulting vapour.

This subset contains only the **operation phase** of the experiments:

- **Six operating points (OP):** OP10, OP17, OP18, OP22, OP29, and OP31. Each represents a different combination of process settings, such as pressure and heat input.
- **13 recordings:** seven normal and six anomalous runs. Each OP has one normal and one anomalous run; OP31 has an additional normal run.
- **One reading per second** from each sensor.
- **YAML annotations** for the anomalous runs, describing disturbances and their timing.

## Folder structure

Each OP has a `measurements/` folder for sensor CSVs and an `annotations/` folder for the anomalous-run YAML. OP31 is shown below; the other OPs follow the same layout without the second normal recording.

```text
batch_distillation_dataset/
  OP10/
  OP17/
  OP18/
  OP22/
  OP29/
  OP31/
    measurements/
      train_normal_experiment_001.csv
      train_normal_experiment_002.csv
      test_anormal_experiment_001.csv
    annotations/
      test_anormal_experiment_001.yaml
```

`normal` denotes a normal run and `anormal` denotes an anomalous run. Match a measurement file to its YAML using the **same OP and filename stem**. For example, the measurement file OP31/measurements/test_anormal_experiment_001.csv has its anomaly annotations in OP31/annotations/test_anormal_experiment_001.yaml.

## Sensor recordings

Each CSV row is one time step. `Time` gives the recording time in `HH:MM:SS` format; the remaining columns contain sensor readings. Each experiment is a separate time series and may have a different start time and duration.

Use the following **11 sensors** for this project. The measurement CSVs also contain additional sensors that need not be included for this project.

| Sensor ID | Measurement |
| --- | --- |
| FT703 | Flow in the top product stream |
| FT704 | Flow in the reflux stream: liquid returned to the column |
| FYI702 | Cooling-water flow signal |
| PDI701 | Pressure difference along the column |
| PDI702 | Pressure-difference measurement associated with buffer vessel V002 |
| PY23 | Pressure in the top vessel G001 |
| T703 | Liquid temperature in the reboiler vessel V001 |
| T705 | Temperature at the top of column section C003 |
| T709 | Temperature below column section C001 |
| T711 | Temperature between column sections C001 and C002 |
| T712 | Temperature between column sections C002 and C003 |

## Anomaly annotations

The YAML files contain an `anomalies` section with event descriptions, timing, and information about affected equipment and sensor observations.

| Field | Meaning |
| --- | --- |
| `id` | Identifier of the event |
| `class` | Annotation status, such as `ConfirmedAnomaly` |
| `hasBeginning` / `hasEnd` | Start and end of the annotated interval |
| `hasCategory` | Anomaly category, such as transient, recurrent, or permanent |
| `Perturbation` | Description of the disturbance |
| `PerturbationMode` | Additional information about the disturbance and its local effects |

Event-level times describe the annotated anomaly interval. Times nested inside `PerturbationMode` describe when the disturbance was applied and may differ from the event-level times.

## Source

This is a subset of the [original Batch Distillation Dataset](https://doi.org/10.5281/zenodo.17395544).
