import pandas as pd
from pathlib import Path
def download_crossword_data():
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    crossword_data_dir = project_root / "crossword_data"

    # Create the crossword_data directory if it doesn't exist
    crossword_data_dir.mkdir(exist_ok=True)

    splits = {'train': 'train.csv', 'validation': 'valid.csv'}
    df = pd.read_csv("hf://datasets/albertxu/CrosswordQA/" + splits["train"])

    print(f"Number of training examples: {len(df)}")
    print(f"Columns in the dataset: {df.columns.tolist()}")
    print("First 5 rows of the dataset:")
    print(df.head())

    # Save to the crossword_data folder with proper path
    output_path = crossword_data_dir / "crossword_train.csv"
    print(f"Saving dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    return





