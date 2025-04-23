import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_perfume_counts(counts_csv, delimiter=",", encoding="latin-1"):
    """
    Load perfume counts data from CSV file.
    
    Args:
        counts_csv: Path to CSV with perfume counts
        delimiter: CSV delimiter
        encoding: File encoding
        
    Returns:
        DataFrame with perfume counts
    """
    logger.info(f"Loading perfume counts from {counts_csv}")
    df_counts = pd.read_csv(counts_csv, delimiter=delimiter, encoding=encoding)
    
    # Ensure we have expected columns
    if 'perfume' not in df_counts.columns or 'count' not in df_counts.columns:
        logger.warning(f"Expected columns 'perfume' and 'count' not found in {counts_csv}")
        
    return df_counts

def get_popular_perfumes_by_count(
    counts_csv,
    perfumes_csv=None,
    gender=None,
    top_k=10,
    counts_delimiter=",",
    perfumes_delimiter=";",
    encoding="latin-1"
):
    """
    Get most popular perfumes based on user count data.
    
    Args:
        counts_csv: Path to CSV with perfume counts
        perfumes_csv: Optional path to perfumes metadata CSV (for gender filtering)
        gender: Filter by gender ('men', 'women', 'unisex' or None for all)
        top_k: Number of perfumes to return
        counts_delimiter: Delimiter for counts CSV
        perfumes_delimiter: Delimiter for perfumes CSV
        encoding: File encoding
        
    Returns:
        List of tuples (perfume_name, count)
    """
    # Load counts data
    df_counts = load_perfume_counts(
        counts_csv=counts_csv,
        delimiter=counts_delimiter,
        encoding=encoding
    )
    
    # If gender is specified and perfumes_csv is provided, filter by gender
    if gender and perfumes_csv:
        logger.info(f"Filtering by gender: {gender}")
        
        # Load perfumes metadata
        df_perfumes = pd.read_csv(perfumes_csv, delimiter=perfumes_delimiter, encoding=encoding)
        
        # Normalize gender values
        gender = gender.lower()
        #print('IN POPULAAR RATING GENDER IS ', gender)
        if gender in ['men','male']:
            filter_gender = 'men'
        elif gender in ['women','female']:
            filter_gender = 'women'
            
        if filter_gender:
            # Get list of perfumes for the specified gender
            gender_perfumes = df_perfumes[
            df_perfumes['Gender'].str.lower().isin([filter_gender.lower(), 'unisex'])]['Perfumes'].tolist()

            
            # Filter counts data to only include perfumes of the specified gender
            df_counts = df_counts[df_counts['perfume'].isin(gender_perfumes)]
    
    # Sort by count and get top k
    top_perfumes = df_counts.sort_values('count', ascending=False).head(top_k)
    
    # Format results
    results = [(row['perfume'], row['count']) for _, row in top_perfumes.iterrows()]
    
    logger.info(f"Found {len(results)} popular perfumes" + (f" for gender={gender}" if gender else ""))
    return results

def get_popular_perfume_recommendations(
    counts_csv,
    perfumes_csv=None,
    gender=None,
    top_k=10,
    counts_delimiter=",",
    perfumes_delimiter=";",
    encoding="latin-1"
):
    """
    Simplified function to get just the perfume names of popular perfumes.
    
    Args:
        counts_csv: Path to CSV with perfume counts
        perfumes_csv: Optional path to perfumes metadata CSV (for gender filtering)
        gender: Filter by gender ('men', 'women', 'unisex' or None for all)
        top_k: Number of perfumes to return
        counts_delimiter: Delimiter for counts CSV
        perfumes_delimiter: Delimiter for perfumes CSV
        encoding: File encoding
        
    Returns:
        List of perfume names
    """
    popular = get_popular_perfumes_by_count(
        counts_csv=counts_csv,
        perfumes_csv=perfumes_csv,
        gender=gender,
        top_k=top_k,
        counts_delimiter=counts_delimiter,
        perfumes_delimiter=perfumes_delimiter,
        encoding=encoding
    )
    return [name for name, _ in popular]
