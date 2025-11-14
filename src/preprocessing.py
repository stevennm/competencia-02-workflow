"""
Data preprocessing module for zLightGBM
Handles data loading and canary feature generation
"""

import polars as pl
import numpy as np
from typing import List, Tuple


def load_data(file_path: str) -> pl.DataFrame:
    """
    Load the dataset from parquet file
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        Polars DataFrame with the dataset
    """
    print(f"Loading dataset from {file_path}...")
    df = pl.read_parquet(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def add_canaritos(df: pl.DataFrame, n_canaritos: int, seed: int = 102191) -> Tuple[pl.DataFrame, List[str]]:
    """
    Add canary features (random noise) at the BEGINNING of the dataset
    
    CRITICAL: Canaries MUST be the first columns (mandatory for zLightGBM)
    
    Canaries are random features with no predictive power. zLightGBM uses them
    as a reference to detect overfitting: if a real feature has less importance
    than a canary, it's a sign of overfitting.
    
    Args:
        df: Input DataFrame
        n_canaritos: Number of canary features to add
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (DataFrame with canaries, list of canary column names)
    """
    print(f"\n{'='*70}")
    print(f"ADDING {n_canaritos} CANARY FEATURES")
    print(f"{'='*70}")
    
    np.random.seed(seed)
    n_rows = df.shape[0]
    
    # Create canary features (uniform random [0, 1])
    canaritos_dict = {}
    for i in range(n_canaritos):
        canaritos_dict[f"canarito_{i+1}"] = np.random.uniform(0, 1, n_rows)
    
    # Create DataFrame with canaries
    df_canaritos = pl.DataFrame(canaritos_dict)
    
    # Concatenate: CANARIES FIRST, then original data
    df_with_canaritos = pl.concat([df_canaritos, df], how="horizontal")
    
    canaritos_names = list(canaritos_dict.keys())
    
    print(f"✓ Original shape: {df.shape}")
    print(f"✓ New shape: {df_with_canaritos.shape}")
    print(f"✓ First 5 columns: {df_with_canaritos.columns[:5]}")
    print(f"✓ Canaries: {canaritos_names[0]}...{canaritos_names[-1]}")
    
    return df_with_canaritos, canaritos_names
