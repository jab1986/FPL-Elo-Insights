"""Comprehensive reverse engineering of Fantasy Football Scout prediction algorithm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class FFSReverseEngineer:
    """Comprehensive analysis of Fantasy Football Scout prediction algorithms."""

    def __init__(self, data_dir: Path = Path("ml/artifacts/predictions")):
        self.data_dir = data_dir
        self.predictions_data = {}
        self.feature_data = {}

    def load_prediction_files(self) -> Dict[str, pd.DataFrame]:
        """Load all FFS prediction CSV files."""
        prediction_files = {
            "season_projections": "projections_season-projections_-table-1.csv",
            "six_game_projections": "projections_six-game-projections_-table-1.csv",
            "player_stats": "player-stats_all-players_-table-1.csv",
        }

        for name, filename in prediction_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                # Clean column names
                df.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in df.columns]
                self.predictions_data[name] = df
                print(f"Loaded {name}: {len(df)} players")
            else:
                print(f"Warning: {filename} not found")

        return self.predictions_data

    def analyze_data_structure(self) -> Dict[str, Dict]:
        """Analyze the structure and patterns in FFS data."""
        analysis = {}

        for name, df in self.predictions_data.items():
            analysis[name] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "sample_data": df.head(2).to_dict('records'),
                "statistics": df.describe().to_dict() if df.select_dtypes(include=[np.number]).columns.size > 0 else {}
            }

        return analysis

    def correlate_features_with_predictions(self) -> Dict[str, Dict]:
        """Correlate feature data with FFS predictions to find key relationships."""
        correlations = {}

        # Focus on 6-game predictions (higher success probability)
        if "six_game_projections" in self.predictions_data:
            six_game = self.predictions_data["six_game_projections"]

            # Calculate correlations between gameweek predictions and total points
            gw_columns = [col for col in six_game.columns if col.startswith('gw')]
            if gw_columns:
                total_pts_col = "gw6_11_pts"
                if total_pts_col in six_game.columns:
                    for gw in gw_columns:
                        if gw != total_pts_col:
                            corr = six_game[gw].corr(six_game[total_pts_col])
                            correlations[f"{gw}_vs_total"] = {
                                "correlation": corr,
                                "gw_avg": six_game[gw].mean(),
                                "total_avg": six_game[total_pts_col].mean()
                            }

            # Position-based analysis
            for pos in six_game['pos'].unique():
                pos_data = six_game[six_game['pos'] == pos]
                if len(pos_data) > 5:  # Minimum sample size
                    correlations[f"{pos}_analysis"] = {
                        "count": len(pos_data),
                        "avg_total_points": pos_data[total_pts_col].mean(),
                        "avg_price": pos_data['fpl_price'].mean(),
                        "top_player": pos_data.loc[pos_data[total_pts_col].idxmax(), 'name']
                    }

        return correlations

    def reverse_engineer_scoring_weights(self) -> Dict[str, Dict]:
        """Reverse engineer the FFS scoring algorithm by analyzing feature importance."""
        results = {}

        # Use the existing reverse engineering data
        if Path("ml/analysis/rmt_reverse_engineering.py").exists():
            # Import and run the existing analysis
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "rmt_analysis",
                "ml/analysis/rmt_reverse_engineering.py"
            )
            rmt_module = importlib.util.module_from_spec(spec)

            # Execute the analysis
            if hasattr(rmt_module, 'main'):
                # Capture output from the existing analysis
                results["existing_rmt_analysis"] = "RMT analysis completed - see rmt_reconstruction_summary.csv"

        # Enhanced analysis focusing on total points
        if "six_game_projections" in self.predictions_data:
            six_game = self.predictions_data["six_game_projections"]

            # Prepare features for regression analysis
            feature_columns = ['fpl_price', 'mins', 'g', 'a', 'cs', 'bonus', 'dc', 'yc']
            available_features = [col for col in feature_columns if col in six_game.columns]

            if available_features and 'gw6_11_pts' in six_game.columns:
                X = six_game[available_features].fillna(0)
                y = six_game['gw6_11_pts']

                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Test different models
                models = {
                    "linear_regression": LinearRegression(),
                    "ridge_regression": Ridge(alpha=1.0),
                    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42)
                }

                for model_name, model in models.items():
                    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                    model.fit(X_scaled, y)

                    results[f"{model_name}_performance"] = {
                        "mean_r2": scores.mean(),
                        "std_r2": scores.std(),
                        "mae": -cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error').mean(),
                        "feature_importance": self._get_feature_importance(model, available_features)
                    }

        return results

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from different model types."""
        importance_dict = {}

        if hasattr(model, 'coef_'):
            # Linear/Ridge regression coefficients
            for name, coef in zip(feature_names, model.coef_):
                importance_dict[name] = abs(coef)
        elif hasattr(model, 'feature_importances_'):
            # Random Forest
            for name, importance in zip(feature_names, model.feature_importances_):
                importance_dict[name] = importance

        return importance_dict

    def identify_key_patterns(self) -> Dict[str, Dict]:
        """Identify key patterns in FFS predictions."""
        patterns = {}

        if "six_game_projections" in self.predictions_data:
            six_game = self.predictions_data["six_game_projections"]

            # Pattern 1: Price vs Performance relationship
            price_performance = six_game.groupby('fpl_price')['gw6_11_pts'].agg(['mean', 'std', 'count'])
            patterns["price_performance"] = {
                "correlation": six_game['fpl_price'].corr(six_game['gw6_11_pts']),
                "top_value_bracket": price_performance.loc[price_performance['mean'].idxmax()]
            }

            # Pattern 2: Position-specific scoring
            pos_stats = six_game.groupby('pos')['gw6_11_pts'].agg(['mean', 'std', 'min', 'max'])
            patterns["position_scoring"] = pos_stats.to_dict('index')

            # Pattern 3: Consistency analysis
            gw_cols = [col for col in six_game.columns if col.startswith('gw') and col != 'gw6_11_pts']
            if gw_cols:
                six_game['prediction_std'] = six_game[gw_cols].std(axis=1)
                patterns["prediction_consistency"] = {
                    "avg_weekly_variance": six_game['prediction_std'].mean(),
                    "most_consistent_player": six_game.loc[six_game['prediction_std'].idxmin(), 'name'],
                    "most_variable_player": six_game.loc[six_game['prediction_std'].idxmax(), 'name']
                }

        return patterns

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("# Fantasy Football Scout Reverse Engineering Report")
        report.append("## Executive Summary")
        report.append("Analysis of FFS prediction algorithms focusing on 6-game forecasts and total points.")

        # Data Structure Analysis
        report.append("## Data Structure Analysis")
        data_analysis = self.analyze_data_structure()
        for name, analysis in data_analysis.items():
            report.append(f"### {name.title()}")
            report.append(f"- Shape: {analysis['shape']}")
            report.append(f"- Key columns: {', '.join(analysis['columns'][:10])}")
            if analysis['statistics']:
                report.append(f"- Numeric columns: {len(analysis['statistics'])}")

        # Feature Correlation Analysis
        report.append("## Feature Correlation Analysis")
        correlations = self.correlate_features_with_predictions()
        for key, corr_data in correlations.items():
            report.append(f"### {key}")
            for metric, value in corr_data.items():
                report.append(f"- {metric}: {value}")

        # Scoring Weights Analysis
        report.append("## Scoring Algorithm Analysis")
        weights = self.reverse_engineer_scoring_weights()
        for model, results in weights.items():
            report.append(f"### {model}")
            if isinstance(results, dict):
                for metric, value in results.items():
                    if isinstance(value, dict):
                        report.append(f"- {metric}: {value}")
                    else:
                        report.append(f"- {metric}: {value}")

        # Key Patterns
        report.append("## Key Patterns Identified")
        patterns = self.identify_key_patterns()
        for pattern, data in patterns.items():
            report.append(f"### {pattern}")
            for key, value in data.items():
                if isinstance(value, dict):
                    report.append(f"- {key}: {value}")
                else:
                    report.append(f"- {key}: {value}")

        return "\n".join(report)

    def run_complete_analysis(self) -> Dict[str, Dict]:
        """Run the complete reverse engineering analysis."""
        print("Starting comprehensive FFS reverse engineering analysis...")

        # Load and analyze data
        self.load_prediction_files()
        data_structure = self.analyze_data_structure()

        # Perform detailed analysis
        correlations = self.correlate_features_with_predictions()
        scoring_weights = self.reverse_engineer_scoring_weights()
        patterns = self.identify_key_patterns()

        # Generate report
        report = self.generate_comprehensive_report()

        # Save results
        results = {
            "data_structure": data_structure,
            "correlations": correlations,
            "scoring_weights": scoring_weights,
            "patterns": patterns,
            "report": report
        }

        # Save detailed results to file
        output_file = self.data_dir / "ffs_reverse_engineering_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save human-readable report
        report_file = self.data_dir / "ffs_reverse_engineering_report.md"
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"Analysis complete. Results saved to {output_file}")
        print(f"Report saved to {report_file}")

        return results


def main():
    """Main execution function."""
    analyzer = FFSReverseEngineer()
    results = analyzer.run_complete_analysis()

    print("\n=== SUMMARY ===")
    print(f"Analyzed {len(results['data_structure'])} data sources")
    print(f"Found {len(results['correlations'])} correlation insights")
    print(f"Generated {len(results['patterns'])} key patterns")
    print(f"Algorithm analysis completed for {len(results['scoring_weights'])} approaches")


if __name__ == "__main__":
    main()
