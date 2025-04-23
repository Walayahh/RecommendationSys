import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_top_rated_perfumes(
    perfumes_csv,
    gender=None,
    top_k=10,
    delimiter=";",
    encoding="latin-1"
):
    """
    Get top perfumes based on weighted rating score.
    
    Args:
        perfumes_csv: Path to perfumes CSV file
        gender: Filter by gender ('men', 'women', 'unisex' or None for all)
        top_k: Number of perfumes to return
        delimiter: CSV delimiter
        encoding: File encoding
        
    Returns:
        List of tuples (perfume_name, weighted_score)
    """
    logger.info(f"Loading perfume data from {perfumes_csv}")
    df_perfumes = pd.read_csv(perfumes_csv, delimiter=delimiter, encoding=encoding)
    
    # Convert rating columns to numeric (handling comma decimal separator)
    df_perfumes['Rating Value'] = pd.to_numeric(
        df_perfumes['Rating Value'].str.replace(',', '.'), 
        errors='coerce'
    )
    df_perfumes['Rating Count'] = pd.to_numeric(df_perfumes['Rating Count'], errors='coerce')
    
    # Filter by gender if specified
    if gender:
        gender = gender.lower()
        if gender in ['men','unisex']:
            df_filtered = df_perfumes[df_perfumes['Gender'].str.lower().isin(['men', 'unisex'])]
        elif gender in ['women','unisex']:
            df_filtered = df_perfumes[df_perfumes['Gender'].str.lower().isin(['women', 'unisex'])]

        else:
            logger.warning(f"Unknown gender: {gender}. Using all perfumes.")
            df_filtered = df_perfumes
    else:
        df_filtered = df_perfumes
    
    # Calculate weighted score (based on IMDb formula)
    # weighted_score = (v/(v+m)) * R + (m/(v+m)) * C
    # where:
    # R = average rating for the perfume
    # v = number of ratings for the perfume
    # m = minimum ratings required (we'll use the 25th percentile)
    # C = mean rating across all perfumes
    
    m = df_filtered['Rating Count'].quantile(0.25)
    C = df_filtered['Rating Value'].mean()
    
    # Calculate weighted score
    def weighted_rating(x, m=m, C=C):
        v = x['Rating Count']
        R = x['Rating Value']
        # Return 0 if null/NaN values
        if pd.isna(v) or pd.isna(R):
            return 0
        return (v/(v+m) * R) + (m/(v+m) * C)
    
    df_filtered['weighted_score'] = df_filtered.apply(weighted_rating, axis=1)
    
    # Sort by weighted score
    top_perfumes = df_filtered.sort_values('weighted_score', ascending=False).head(top_k)
    
    # Format results
    results = [(row['Perfumes'], row['weighted_score']) for _, row in top_perfumes.iterrows()]
    
    logger.info(f"Found {len(results)} top rated perfumes for gender={gender}")
    return results

def get_top_rated_recommendations(
    perfumes_csv,
    gender=None,
    top_k=10,
    delimiter=";",
    encoding="latin-1"
):
    """
    Simplified function to get just the perfume names of top rated perfumes.
    
    Args:
        perfumes_csv: Path to perfumes CSV file
        gender: Filter by gender ('men', 'women', 'unisex' or None for all)
        top_k: Number of perfumes to return
        delimiter: CSV delimiter
        encoding: File encoding
        
    Returns:
        List of perfume names
    """
    top_rated = get_top_rated_perfumes(
        perfumes_csv=perfumes_csv,
        gender=gender,
        top_k=top_k,
        delimiter=delimiter,
        encoding=encoding
    )
    return [name for name, _ in top_rated]

