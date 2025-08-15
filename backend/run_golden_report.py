#!/usr/bin/env python3
"""
Golden Run Report Runner
=======================

Simple script to generate golden run reports for different quarters.
"""

import os
import sys
from datetime import date, datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reports.golden_run_reporter import GoldenRunReporter
from utils.logger import setup_logging

logger = setup_logging()

def main():
    """Main function to run golden run reports"""
    
    print("=== Swing Trading Golden Run Reporter ===")
    print()
    
    # Initialize reporter
    reporter = GoldenRunReporter(output_dir="reports/output")
    
    try:
        # Generate report for last quarter
        print("Generating golden run report for the most recent completed quarter...")
        report_data = reporter.generate_quarterly_report()
        
        if 'error' in report_data:
            print(f"❌ Error generating report: {report_data['error']}")
            print()
            print("Note: This is expected if no recommendations exist in the database.")
            print("To test with sample data, you can:")
            print("1. Run the analysis system to generate some recommendations")
            print("2. Or modify the reporter to use sample data for demonstration")
            return
        
        # Print summary
        period = report_data.get('period', {})
        summary = report_data.get('summary_stats', {})
        
        print("✅ Golden run report generated successfully!")
        print()
        print("📊 Report Summary:")
        print(f"   Period: {period.get('start', 'Unknown')} to {period.get('end', 'Unknown')}")
        print(f"   Quarter: {period.get('quarter', 'Unknown')}")
        print(f"   Total Recommendations: {summary.get('total_recommendations', 0)}")
        print(f"   Average Combined Score: {summary.get('score_statistics', {}).get('combined', {}).get('mean', 0):.3f}")
        
        # Print precision highlights
        precision_data = report_data.get('precision_analysis', {})
        if precision_data:
            print()
            print("🎯 Precision Highlights:")
            for horizon, data in list(precision_data.items())[:2]:  # Show first 2 horizons
                for threshold, metrics in list(data.items())[:2]:  # Show first 2 thresholds
                    precision = metrics.get('precision', 0)
                    print(f"   {horizon} @ {threshold}: {precision:.1%} precision ({metrics.get('successful_picks', 0)}/{metrics.get('total_picks', 0)})")
        
        # Print MAE highlights
        mae_data = report_data.get('mae_analysis', {})
        if mae_data and 'overall_statistics' in mae_data:
            mae_stats = mae_data['overall_statistics']
            print()
            print("📉 MAE Highlights:")
            print(f"   Average MAE: {mae_stats.get('mean_mae', 0):.1%}")
            print(f"   Median MAE: {mae_stats.get('median_mae', 0):.1%}")
            print(f"   95th Percentile MAE: {mae_stats.get('percentiles', {}).get('95th', 0):.1%}")
        
        # Print file locations
        print()
        print("📁 Report Files:")
        print(f"   HTML Report: {reporter.output_dir}/golden_run_report_{period.get('quarter', 'unknown').replace('-', '_')}.html")
        print(f"   Charts: {reporter.output_dir}/")
        
        # Print next steps
        print()
        print("🚀 Next Steps:")
        print("1. Open the HTML report in your browser to view detailed analysis")
        print("2. Review precision rates and consider adjusting thresholds if needed")
        print("3. Analyze MAE patterns to optimize stop-loss strategies")
        print("4. Check sector concentration and adjust filters if necessary")
        print("5. Compare gate pass rates to identify potential improvements")
        
    except Exception as e:
        logger.error(f"Error running golden run reporter: {e}")
        print(f"❌ Error: {e}")
        print()
        print("This might be due to:")
        print("1. Database connection issues")
        print("2. Missing dependencies (matplotlib, seaborn)")
        print("3. Insufficient data in the database")
        
        # Show how to install dependencies
        print()
        print("To install missing dependencies:")
        print("pip install matplotlib seaborn")

if __name__ == "__main__":
    main()
