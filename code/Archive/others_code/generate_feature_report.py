"""
Feature Report Generator
Creates comprehensive reports of input features used across all models and sites

Author: Dr. Masrur
Last Updated: 2026-01-05
"""

import os
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_consolidated_feature_report(output_directory):
    """
    Generate consolidated feature report from all input_features folders
    
    Args:
        output_directory:  Base output directory containing model results
    """
    
    logging.info("="*80)
    logging.info("GENERATING CONSOLIDATED FEATURE REPORT")
    logging.info("="*80)
    
    # Find all input_features directories
    input_features_dirs = []
    for root, dirs, files in os.walk(output_directory):
        if 'input_features' in dirs:
            input_features_dirs.append(os.path.join(root, 'input_features'))
    
    if not input_features_dirs:
        logging.warning("No input_features directories found")
        return
    
    logging.info(f"Found {len(input_features_dirs)} input_features directories")
    
    # Collect all feature summary files
    all_summaries = []
    all_detailed_features = []
    
    for input_features_dir in input_features_dirs:
        model_name = os.path.basename(os.path.dirname(input_features_dir))
        
        for filename in os.listdir(input_features_dir):
            filepath = os.path.join(input_features_dir, filename)
            
            # Process summary files
            if 'features_summary_all_missingness.csv' in filename:
                try:
                    df = pd.read_csv(filepath)
                    df['Model_Directory'] = model_name
                    all_summaries.append(df)
                    logging.info(f"Loaded summary:  {filename}")
                except Exception as e:
                    logging.error(f"Error loading {filename}: {e}")
            
            # Process detailed feature files
            elif 'features_detailed' in filename:
                try: 
                    df = pd.read_csv(filepath)
                    # Extract metadata from filename
                    parts = filename.replace('. csv', '').split('_')
                    site = parts[0]
                    target = parts[1]
                    model = parts[2]
                    missingness = parts[-1]
                    
                    df['Site'] = site
                    df['Target'] = target
                    df['Model'] = model
                    df['Missingness'] = missingness
                    df['Model_Directory'] = model_name
                    
                    all_detailed_features. append(df)
                except Exception as e:
                    logging.debug(f"Error loading {filename}: {e}")
    
    # Create consolidated reports directory
    consolidated_dir = os.path.join(output_directory, 'Consolidated_Feature_Reports')
    os.makedirs(consolidated_dir, exist_ok=True)
    
    # Save consolidated summary report
    if all_summaries:
        consolidated_summary = pd.concat(all_summaries, ignore_index=True)
        
        # Sort by Model, Site, Missingness
        consolidated_summary = consolidated_summary.sort_values(
            ['Model', 'Site', 'Missingness_Level']
        )
        
        summary_output = os.path.join(consolidated_dir, 'All_Models_Feature_Summary.csv')
        consolidated_summary.to_csv(summary_output, index=False)
        logging.info(f"\n✅ Saved consolidated summary:  {summary_output}")
        
        # Create pivot table for easy comparison
        pivot_features = consolidated_summary.pivot_table(
            index=['Site', 'Target_Column'],
            columns='Model',
            values='Total_Features',
            aggfunc='first'
        )
        
        pivot_output = os.path.join(consolidated_dir, 'Feature_Count_Comparison_by_Model.csv')
        pivot_features.to_csv(pivot_output)
        logging.info(f"✅ Saved feature comparison: {pivot_output}")
        
        # Create summary statistics
        summary_stats = consolidated_summary.groupby('Model').agg({
            'Total_Features': ['mean', 'min', 'max'],
            'Base_Features_Count': ['mean', 'min', 'max'],
            'Spatial_Features_Count': ['mean', 'min', 'max'],
            'Temporal_Features_Count': ['mean', 'min', 'max'],
        }).round(2)
        
        stats_output = os.path.join(consolidated_dir, 'Feature_Statistics_by_Model.csv')
        summary_stats.to_csv(stats_output)
        logging.info(f"✅ Saved feature statistics: {stats_output}")
    
    # Save consolidated detailed features report
    if all_detailed_features:
        consolidated_detailed = pd.concat(all_detailed_features, ignore_index=True)
        
        detailed_output = os.path.join(consolidated_dir, 'All_Models_Detailed_Features. csv')
        consolidated_detailed. to_csv(detailed_output, index=False)
        logging.info(f"✅ Saved consolidated detailed features: {detailed_output}")
        
        # Create feature frequency report
        feature_frequency = consolidated_detailed.groupby(['Feature_Name', 'Feature_Type']).size().reset_index(name='Usage_Count')
        feature_frequency = feature_frequency.sort_values('Usage_Count', ascending=False)
        
        frequency_output = os.path.join(consolidated_dir, 'Feature_Usage_Frequency.csv')
        feature_frequency.to_csv(frequency_output, index=False)
        logging.info(f"✅ Saved feature usage frequency: {frequency_output}")
    
    # Generate summary statistics
    logging.info("\n" + "="*80)
    logging.info("FEATURE REPORT SUMMARY")
    logging.info("="*80)
    
    if all_summaries:
        total_runs = len(consolidated_summary)
        unique_models = consolidated_summary['Model'].nunique()
        unique_sites = consolidated_summary['Site'].nunique()
        
        logging.info(f"Total imputation runs: {total_runs}")
        logging.info(f"Unique models: {unique_models}")
        logging.info(f"Unique sites: {unique_sites}")
        logging.info(f"\nFeature counts across all runs:")
        logging.info(f"  Average total features: {consolidated_summary['Total_Features'].mean():.1f}")
        logging.info(f"  Average base features: {consolidated_summary['Base_Features_Count'].mean():.1f}")
        logging.info(f"  Average spatial features: {consolidated_summary['Spatial_Features_Count'].mean():.1f}")
        logging.info(f"  Average temporal features:  {consolidated_summary['Temporal_Features_Count'].mean():.1f}")
    
    logging.info(f"\nReports saved to: {consolidated_dir}")
    logging.info("="*80)


if __name__ == "__main__": 
    import sys
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        # Use config
        try:
            import config_spatial as config
            output_dir = config.OUTPUT_DIRECTORY
        except: 
            output_dir = "./output"
    
    generate_consolidated_feature_report(output_dir)