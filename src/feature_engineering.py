import numpy as np
import pandas as pd

# Define the feature engineering function BEFORE using it
def feature_engineering(df):
    # 1. Log transform skewed numeric features to reduce skewness
    if 'SalePrice' in df.columns:
        df['LogSalePrice'] = np.log1p(df['SalePrice'])

    # Apply log transform on GrLivArea to handle skewness
    df['LogGrLivArea'] = np.log1p(df['GrLivArea'])

    # 2. Create total square footage feature
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # 3. Create house age and renovation age
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    # 4. Create binary features for presence of pool and fireplace
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

    # 5. Encode categorical ordinal features (example for OverallQual)
    quality_map = {
        1: 'Very Poor', 2: 'Poor', 3: 'Fair', 4: 'Below Average', 5: 'Average',
        6: 'Above Average', 7: 'Good', 8: 'Very Good', 9: 'Excellent', 10: 'Very Excellent'
    }
    if 'OverallQual' in df.columns:
        df['OverallQualCat'] = df['OverallQual'].map(quality_map)

    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df

if __name__ == "__main__":
    import pandas as pd
    train_df = pd.read_csv('D:/HousePricePredict/data/processed/train_clean.csv')
    test_df = pd.read_csv('D:/HousePricePredict/data/processed/test_clean.csv')

    train_df_feat = feature_engineering(train_df)
    test_df_feat = feature_engineering(test_df)

    train_df_feat.to_csv('D:/HousePricePredict/data/processed/train_feat.csv', index=False)
    test_df_feat.to_csv('D:/HousePricePredict/data/processed/test_feat.csv', index=False)

