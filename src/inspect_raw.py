
import pandas as pd

def inspect_raw_values():
    filepath = 'data/survey_data.xlsx'
    try:
        # Read as string to see actual text
        df = pd.read_excel(filepath, dtype=str)
        

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        


        target_cols = [
            'Gender',
            'Location'
        ]
        
        for col in target_cols:
            if col in df.columns:
                print(f"\nColumn: {col}")
                print(f"Unique Values: {df[col].unique()}")
            else:
                print(f"\nColumn '{col}' NOT FOUND. Available columns: {df.columns.tolist()}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_raw_values()
