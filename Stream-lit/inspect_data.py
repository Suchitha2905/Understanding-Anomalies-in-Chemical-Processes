from pathlib import Path
import pandas as pd
import yaml

DATA_DIR = Path(__file__).parent / "batch_distillation_dataset"

operating_points = ["OP10", "OP17", "OP18", "OP22", "OP29", "OP31"]

for op in operating_points:
    print("\n" + "=" * 60)
    print(f"OPERATING POINT: {op}")
    print("=" * 60)

    op_dir = DATA_DIR / op

    measurement_dir = op_dir / "measurements"
    annotation_dir = op_dir / "annotations"

    for csv_file in sorted(measurement_dir.glob("*.csv")):
        print(f"\nFILE: {csv_file.name}")

        df = pd.read_csv(csv_file)

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

    for yaml_file in sorted(annotation_dir.glob("*.yaml")):
        print(f"\nANNOTATION FILE: {yaml_file.name}")

        with open(yaml_file, "r", encoding="utf-8") as file:
            annotations = yaml.safe_load(file)

        print("YAML structure:")
        print(str(annotations)[:3000])