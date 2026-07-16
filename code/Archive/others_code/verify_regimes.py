import pandas as pd
import os

def verify_regime_differences(output_dir, model_name, site, target):
    """Check if different regimes produce different results"""
    
    regimes = ['random', 'short_gap', 'medium_gap', 'long_gap', 'event']
    
    print(f"\n{'='*80}")
    print(f"VERIFYING:  {model_name} | {site} | {target}")
    print(f"{'='*80}\n")
    
    results = {}
    
    for regime in regimes:
        metrics_file = f"{output_dir}/{model_name}/{regime}/metrics/{site}_{target}_{model_name}_{site}_{regime}_all_metrics.csv"
        
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            results[regime] = df[['Missingness', 'Root Mean Squared Error (RMSE)', 'Correlation Coefficient (R)']].copy()
            print(f"✅ {regime: 15s} | RMSE: {df['Root Mean Squared Error (RMSE)'].mean():.4f} | R: {df['Correlation Coefficient (R)'].mean():.4f}")
        else:
            print(f"❌ {regime:15s} | FILE NOT FOUND")
    
    # Check if results are different
    if len(results) > 1:
        rmse_values = [df['Root Mean Squared Error (RMSE)'].mean() for df in results.values()]
        if len(set([f"{v:.4f}" for v in rmse_values])) == 1:
            print("\n⚠️  WARNING: All regimes have IDENTICAL RMSE!")
        else:
            print("\n✅ Results are DIFFERENT across regimes (expected)")

if __name__ == "__main__":
    verify_regime_differences(
        output_dir="Imputation_Result_Spatial_Temporal_V7",
        model_name="LightGBM",
        site="ARMIDALE",
        target="PM2.5"
    )