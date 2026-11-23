import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# Açıklama: Witcher karakter veri setini CSV'den okur ve bir pandas.DataFrame döndürür.
# Input:filepath: str (CSV dosya yolu)
# Output: pd.DataFrame
def load_witcher_dataset(filepath):
    return pd.read_csv(filepath)

# Açıklama: Hedef sınıf kolonunun dağılımını verir.
# Input: df: pd.DataFrame, target_column: str
# Output: dict (sınıf: adet)
def get_class_distribution(df, target_column):
    return df[target_column].value_counts().to_dict()

# Açıklama: Belirtilen kategorik kolona Label Encoding uygular.
# Input: df: pd.DataFrame, column: str
# Output: pd.DataFrame (kolon Label Encoding yapılmış)
def apply_label_encoding(df, column):
   df[column] = LabelEncoder().fit_transform(df[column])
   return df


# Açıklama: Belirtilen kategorik kolona One Hot Encoding uygular.
# Input: df: pd.DataFrame, column: str
# Output: pd.DataFrame (kolon OHE yapılmış)
def apply_one_hot_encoding(df, column):
     return pd.get_dummies(df, columns=[column], dtype=int)

# Açıklama: Hedef kolonun dengesiz sınıflarını eşitlemek için down sampling uygular.
# Input: df: pd.DataFrame, target_column: str
# Output: pd.DataFrame (dengelenmiş)
def down_sample(df, target_column):
    n = df[target_column].value_counts().min()
    return df.groupby(target_column).sample(n=n, random_state=42)


# Açıklama: Hedef kolonun dengesiz sınıflarını eşitlemek için up sampling uygular.
# Input: df: pd.DataFrame, target_column: str
# Output: pd.DataFrame
def up_sample(df, target_column):
    n = df[target_column].value_counts().max()
    return df.groupby(target_column).sample(n=n, replace=True, random_state=42)

# Açıklama: Girdi verisine SMOTE uygular ve yeni X, y döndürür.
# Input: X: np.ndarray veya pd.DataFrame, y: np.ndarray veya pd.Series
# Output:(X_resampled, y_resampled) (numpy array)
def apply_smote(X, y):
     return SMOTE(random_state=42).fit_resample(np.asarray(X), np.asarray(y))

# Açıklama: DataFrame'i özellikler (X) ve hedef (y) olarak ayırır.
# Input: df: pd.DataFrame, target_column: str
# Output:(X: pd.DataFrame, y: pd.Series)
def split_features_target(df, target_column):
   return df.drop(columns=[target_column]), df[target_column]

# Açıklama: Veri setinin temel istatistiklerini ve kolon bilgilerini döndürür.
# Input: df: pd.DataFrame
# Output: dict
def summarize_dataset(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "describe": df.describe(include="all").to_dict()
    }


