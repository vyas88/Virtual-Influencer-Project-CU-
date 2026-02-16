
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer

def load_data(filepath='data/survey_data.xlsx'):
    """
    Load the dataset from an Excel file.
    """
    try:
        df = pd.read_excel(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def preprocess_data(df):
    """
    Preprocess the data: handle missing values, encode categorical variables, normalize numerical features.
    """
    # 1. Clean Column Names
    df.columns = df.columns.str.strip()
    
    # 2. Rename columns for easier access
    column_mapping = {
        'Age Group': 'age',
        'Gender': 'gender',
        'Location': 'location',
        'Occupation': 'occupation',
        'How frequently do you use social media?': 'social_media_usage',
        'How often do you engage with virtual influencers on social media?': 'vi_engagement_freq',
        'How realistic do you find the personalities of virtual influencers?': 'vi_realism',
        'Do you trust virtual influencers as much as human influencers when they promote a product or brand?': 'vi_trust',
        'Overall, how satisfied are you with virtual influencers compared to human influencers?': 'vi_satisfaction',
        'Would you consider buying a product recommended by a virtual influencer?': 'vi_purchase_intent',
        'Are you familiar with virtual influencers (AI-generated influencers)?': 'vi_familiarity'
    }
    df = df.rename(columns=column_mapping)
    
    # 3. Handle Target Variable (Satisfaction)
    # Map text responses to numbers if they are text. Inspecting the data suggests they might be strings.
    # We will try to map standardized Likert scales if possible, or use Label Encoding.
    # For now, let's look at unique values for the target. 
    # Since we can't interactively check, we will blindly attempt to clean common Likert text.
    
    # Mapping not needed as inspection showed data is already numeric (1-5 scales)
    # Keeping logic simple.
    pass

    # Apply mapping to target and potential numerical features
    # TARGETS for Multi-Output: Satisfaction, Trust, Engagement, Purchase Intent
    target_cols = ['vi_satisfaction', 'vi_trust', 'vi_engagement_freq', 'vi_purchase_intent']
    
    # If any target is missing, drop row
    df = df.dropna(subset=target_cols)
    
    # Attempt to map known Likert columns using a generic apply map where possible and fallback to label encoding
    numerical_cols = ['vi_engagement_freq', 'vi_realism', 'vi_trust', 'vi_purchase_intent']
    
    # Combined list of columns to clean (ensure uniqueness if overlap)
    cols_to_clean = list(set(numerical_cols + target_cols))
    
    for col in cols_to_clean:
        if col in df.columns:
             unique_vals = df[col].unique()
             print(f"Column {col} raw unique values: {unique_vals}")
             
             # If already numeric, skip mapping (or just ensure type)
             if pd.api.types.is_numeric_dtype(df[col]):
                 print(f"Column {col} is already numeric. Skipping map.")
                 continue
                 
             # Ensure string and strip whitespace
             if df[col].dtype == 'object':
                 df[col] = df[col].astype(str).str.strip()
             
             # Convert using map (Skipping as we know they are numeric now)
             # df[col] = df[col].map(likert_mapping)
             
             # debug print
             print(f"Column {col} unique values after map: {df[col].unique()}")
             
             # If mapping failed (still object or NaNs), use LabelEncoder or converting to numeric errors='coerce'
             df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaNs in numerical columns with median
    for col in cols_to_clean:
        if col in df.columns:
             df[col] = df[col].fillna(df[col].median())

    # Add Interaction Feature: Age * Social Media Usage
    # Assuming standard ordinal direction (Higher Age * Higher Usage = ??? but useful for non-linear detection)
    if 'age' in df.columns and 'social_media_usage' in df.columns:
        # Ensure they are numeric first (they should be based on inspection)
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
        df['social_media_usage'] = pd.to_numeric(df['social_media_usage'], errors='coerce').fillna(0)
        
        df['age_x_usage'] = df['age'] * df['social_media_usage']

    # 4. Handle Categorical Features (One-Hot Encoding)
    # Based on inspection, Age and Social Media Usage are ordinal (1-5), so we treat them as numeric.
    # Gender, Location, Occupation are categorical codes (1, 2, 3...), so we One-Hot Encode them.
    categorical_cols = ['gender', 'location', 'occupation']
    
    # Filter only columns that exist
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 5. Drop Text/Multi-select columns for MVP (Too complex to parse without more inspection)
    # Ideally we'd parse the multi-selects, but let's stick to core features for the NN first.
    # Drop columns that are still 'object' type (likely text)
    df_numeric = df_encoded.select_dtypes(include=[np.number])
    
    # Ensure target is present
    # Block removed as we check target_cols in the loop below
    pass

    # Drop columns with all NaNs
    df_numeric = df_numeric.dropna(axis=1, how='all')

    # Drop columns with single unique value (zero variance), BUT KEEP TARGETS
    nunique = df_numeric.nunique()
    cols_to_drop = nunique[nunique <= 1].index
    # Ensure targets are not dropped
    cols_to_drop = [c for c in cols_to_drop if c not in target_cols]
    df_numeric = df_numeric.drop(cols_to_drop, axis=1)
    
    print(f"Columns after cleaning: {df_numeric.columns.tolist()}")
    for t in target_cols:
        if t not in df_numeric.columns:
            print(f"Target {t} lost!")

    return df_numeric, target_cols

def get_train_test_data(df_numeric, target_cols):
    """
    Split the data into training and testing sets for MULTI-OUTPUT.
    """
    X = df_numeric.drop(target_cols, axis=1)
    y = df_numeric[target_cols]
    
    # Drop columns with zero variance to avoid scaling errors
    # (Handling again just in case splitting caused it)
    # X = X.loc[:, X.nunique() > 1] # Better done before split? No, do it before.
    
    # Standardize features
    scaler = StandardScaler()
    # Handle case where specific columns might still cause issues? 
    # StandardScaler handles constant features by setting them to 0 if with_std=False usually, but let's be safe.
    
    try:
        X_scaled = scaler.fit_transform(X)
    except ValueError as e:
        print(f"Scaling error: {e}")
        # Fallback: fill NaNs with 0 if scaling failed
        X_scaled = X.fillna(0).values

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print("Columns:", df.columns)
        processed_df, target = preprocess_data(df)
        print("Processed shape:", processed_df.shape)
        X_train, X_test, y_train, y_test, _ = get_train_test_data(processed_df, target)
        print("Train shape:", X_train.shape)
