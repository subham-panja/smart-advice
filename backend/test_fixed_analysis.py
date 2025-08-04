#!/usr/bin/env python3
"""
Test Script for Fixed Stock Analysis
File: test_fixed_analysis.py

This script tests the fixed analysis with 200 stocks and validates the results.
"""

import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from run_analysis import AutomatedStockAnalysis
from database import get_mongodb
from utils.logger import setup_logging

logger = setup_logging()

def validate_recommendation_data(recommendation):
    """
    Validate a single recommendation for data quality issues.
    
    Args:
        recommendation: MongoDB document representing a recommendation
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check buy_price
    buy_price = recommendation.get('buy_price', 0)
    if buy_price == 0:
        issues.append("buy_price is 0")
    
    # Check sell_price
    sell_price = recommendation.get('sell_price', 0)
    if sell_price == 0:
        issues.append("sell_price is 0")
    
    # Check est_time_to_target
    est_time_to_target = recommendation.get('est_time_to_target', 'Unknown')
    if est_time_to_target == 'Unknown':
        issues.append("est_time_to_target is 'Unknown'")
    
    # Check scores
    technical_score = recommendation.get('technical_score', 0)
    fundamental_score = recommendation.get('fundamental_score', 0)
    sentiment_score = recommendation.get('sentiment_score', 0)
    
    if technical_score == 0:
        issues.append("technical_score is 0")
    if fundamental_score == 0:
        issues.append("fundamental_score is 0")
    if sentiment_score == 0:
        issues.append("sentiment_score is 0")
    
    return {
        'symbol': recommendation.get('symbol', 'UNKNOWN'),
        'issues': issues,
        'has_issues': len(issues) > 0,
        'buy_price': buy_price,
        'sell_price': sell_price,
        'est_time_to_target': est_time_to_target,
        'technical_score': technical_score,
        'fundamental_score': fundamental_score,
        'sentiment_score': sentiment_score
    }

def run_test_analysis():
    """Run analysis on 200 stocks and validate results."""
    logger.info("Starting test analysis with 200 stocks...")
    
    try:
        # Create analyzer instance
        analyzer = AutomatedStockAnalysis()
        
        # Set test configuration
        analyzer.app.config['DATA_PURGE_DAYS'] = 0  # Purge all old data
        
        # Run analysis with 200 stocks
        logger.info("Running analysis on 200 stocks...")
        analyzer.run_analysis(max_stocks=200, use_all_symbols=False)  # Use filtered symbols
        
        logger.info("Analysis completed. Validating results...")
        
        # Check results in database
        with analyzer.app.app_context():
            db = get_mongodb()
            
            # Get all recommendations
            recommendations = list(db.recommended_shares.find({}))
            total_recommendations = len(recommendations)
            
            logger.info(f"Found {total_recommendations} recommendations in database")
            
            if total_recommendations == 0:
                logger.error("No recommendations found in database!")
                return False
            
            # Validate each recommendation
            validation_results = []
            issues_count = 0
            
            for rec in recommendations:
                validation = validate_recommendation_data(rec)
                validation_results.append(validation)
                
                if validation['has_issues']:
                    issues_count += 1
                    logger.warning(f"Issues found in {validation['symbol']}: {', '.join(validation['issues'])}")
                else:
                    logger.info(f"✓ {validation['symbol']}: buy=${validation['buy_price']:.2f}, "
                               f"sell=${validation['sell_price']:.2f}, eta={validation['est_time_to_target']}, "
                               f"scores: tech={validation['technical_score']:.3f}, "
                               f"fund={validation['fundamental_score']:.3f}, "
                               f"sent={validation['sentiment_score']:.3f}")
            
            # Print summary
            success_count = total_recommendations - issues_count
            success_rate = (success_count / total_recommendations) * 100 if total_recommendations > 0 else 0
            
            logger.info(f"\n=== VALIDATION SUMMARY ===")
            logger.info(f"Total recommendations: {total_recommendations}")
            logger.info(f"Recommendations with issues: {issues_count}")
            logger.info(f"Successful recommendations: {success_count}")
            logger.info(f"Success rate: {success_rate:.1f}%")
            
            # Detailed issue breakdown
            issue_types = {}
            for validation in validation_results:
                for issue in validation['issues']:
                    issue_types[issue] = issue_types.get(issue, 0) + 1
            
            if issue_types:
                logger.info(f"\n=== ISSUE BREAKDOWN ===")
                for issue, count in issue_types.items():
                    percentage = (count / total_recommendations) * 100
                    logger.info(f"{issue}: {count} occurrences ({percentage:.1f}%)")
            
            # Check if we need to re-run analysis
            if success_rate < 80:
                logger.error(f"Success rate {success_rate:.1f}% is below 80% threshold. Analysis needs improvement.")
                return False
            else:
                logger.info(f"✓ Success rate {success_rate:.1f}% meets the 80% threshold!")
                return True
    
    except Exception as e:
        logger.error(f"Error in test analysis: {e}")
        return False

def main():
    """Main entry point."""
    logger.info("=== FIXED STOCK ANALYSIS TEST ===")
    logger.info(f"Test started at: {datetime.now()}")
    
    success = run_test_analysis()
    
    logger.info(f"Test completed at: {datetime.now()}")
    
    if success:
        logger.info("✓ TEST PASSED: Fixed analysis is working correctly!")
        return 0
    else:
        logger.error("✗ TEST FAILED: Analysis still has issues that need fixing!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
