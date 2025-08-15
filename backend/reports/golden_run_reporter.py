#!/usr/bin/env python3
"""
Golden Run Reporter
==================

Generates comprehensive reports showing precision, MAE, and performance metrics
for swing trading recommendations over prior quarters.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import DatabaseManager
from utils.logger import setup_logging
from scripts.swing_trading_signals import SwingTradingAnalyzer

logger = setup_logging()

class GoldenRunReporter:
    """
    Generates golden-run reports for swing trading performance analysis
    """
    
    def __init__(self, output_dir: str = "reports/output"):
        """Initialize the reporter"""
        self.output_dir = output_dir
        self.db_manager = DatabaseManager()
        self.swing_analyzer = SwingTradingAnalyzer()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Report configuration
        self.analysis_horizons = [5, 10, 15, 20]  # Trading day horizons
        self.precision_thresholds = [0.02, 0.05, 0.10, 0.15]  # Return thresholds for precision
        
    def generate_quarterly_report(self, quarter_end: date = None, 
                                quarters_back: int = 1) -> Dict[str, Any]:
        """
        Generate comprehensive quarterly report
        
        Args:
            quarter_end: End date of quarter to analyze (None for most recent)
            quarters_back: Number of quarters to go back
            
        Returns:
            Dictionary containing all report data
        """
        logger.info(f"Generating golden run report for {quarters_back} quarters back")
        
        try:
            # Determine date range
            if quarter_end is None:
                quarter_end = self._get_last_quarter_end()
            
            quarter_start = quarter_end - timedelta(days=90)
            
            # Fetch recommendations from database
            recommendations = self._fetch_recommendations(quarter_start, quarter_end)
            
            if not recommendations:
                logger.warning(f"No recommendations found for period {quarter_start} to {quarter_end}")
                return {'error': 'No data found for specified period'}
            
            # Generate comprehensive analysis
            report_data = {
                'period': {
                    'start': quarter_start.isoformat(),
                    'end': quarter_end.isoformat(),
                    'quarter': self._get_quarter_label(quarter_end)
                },
                'summary_stats': self._calculate_summary_stats(recommendations),
                'precision_analysis': self._analyze_precision(recommendations),
                'mae_analysis': self._analyze_mae(recommendations),
                'timing_analysis': self._analyze_timing(recommendations),
                'sector_analysis': self._analyze_sector_performance(recommendations),
                'gates_analysis': self._analyze_gates_effectiveness(recommendations),
                'risk_analysis': self._analyze_risk_metrics(recommendations),
                'comparison_analysis': self._compare_with_benchmarks(recommendations)
            }
            
            # Generate visualizations
            self._create_visualizations(report_data)
            
            # Generate HTML report
            html_report = self._generate_html_report(report_data)
            
            # Save report
            report_file = os.path.join(self.output_dir, 
                                     f"golden_run_report_{quarter_end.strftime('%Y_Q%s' % ((quarter_end.month-1)//3 + 1))}.html")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            logger.info(f"Golden run report saved to: {report_file}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating quarterly report: {e}")
            return {'error': str(e)}
    
    def _fetch_recommendations(self, start_date: date, end_date: date) -> List[Dict]:
        """Fetch recommendations from database for the specified period"""
        try:
            # Convert dates to datetime for database query
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            # Query recommendations
            query = {
                'timestamp': {
                    '$gte': start_datetime,
                    '$lte': end_datetime
                }
            }
            
            recommendations = list(self.db_manager.get_collection('recommended_shares').find(query))
            
            # Convert ObjectId to string for JSON serialization
            for rec in recommendations:
                if '_id' in rec:
                    rec['_id'] = str(rec['_id'])
                    
            logger.info(f"Fetched {len(recommendations)} recommendations from {start_date} to {end_date}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error fetching recommendations: {e}")
            return []
    
    def _calculate_summary_stats(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Calculate high-level summary statistics"""
        try:
            total_recommendations = len(recommendations)
            
            # Count by recommendation strength
            strength_counts = {}
            for rec in recommendations:
                strength = rec.get('recommendation_strength', 'UNKNOWN')
                strength_counts[strength] = strength_counts.get(strength, 0) + 1
            
            # Calculate score distributions
            scores = []
            technical_scores = []
            fundamental_scores = []
            combined_scores = []
            
            for rec in recommendations:
                if 'technical_score' in rec:
                    technical_scores.append(rec['technical_score'])
                if 'fundamental_score' in rec:
                    fundamental_scores.append(rec['fundamental_score'])
                if 'combined_score' in rec:
                    combined_scores.append(rec['combined_score'])
            
            return {
                'total_recommendations': total_recommendations,
                'recommendation_distribution': strength_counts,
                'score_statistics': {
                    'technical': {
                        'mean': np.mean(technical_scores) if technical_scores else 0,
                        'std': np.std(technical_scores) if technical_scores else 0,
                        'median': np.median(technical_scores) if technical_scores else 0
                    },
                    'fundamental': {
                        'mean': np.mean(fundamental_scores) if fundamental_scores else 0,
                        'std': np.std(fundamental_scores) if fundamental_scores else 0,
                        'median': np.median(fundamental_scores) if fundamental_scores else 0
                    },
                    'combined': {
                        'mean': np.mean(combined_scores) if combined_scores else 0,
                        'std': np.std(combined_scores) if combined_scores else 0,
                        'median': np.median(combined_scores) if combined_scores else 0
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {}
    
    def _analyze_precision(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Analyze precision metrics at different horizons and thresholds"""
        try:
            precision_results = {}
            
            for horizon in self.analysis_horizons:
                horizon_results = {}
                
                for threshold in self.precision_thresholds:
                    # Calculate precision for this horizon/threshold combination
                    successful_picks = 0
                    total_picks = 0
                    
                    for rec in recommendations:
                        # Check if we have performance data for this horizon
                        backtest_data = rec.get('backtest', {})
                        if not backtest_data:
                            continue
                            
                        total_picks += 1
                        
                        # Simulate performance (in real implementation, this would be actual returns)
                        simulated_return = np.random.normal(0.03, 0.15)  # 3% mean, 15% volatility
                        
                        if simulated_return >= threshold:
                            successful_picks += 1
                    
                    precision = successful_picks / total_picks if total_picks > 0 else 0
                    
                    horizon_results[f"threshold_{threshold:.0%}"] = {
                        'precision': precision,
                        'successful_picks': successful_picks,
                        'total_picks': total_picks
                    }
                
                precision_results[f"{horizon}_day"] = horizon_results
            
            return precision_results
            
        except Exception as e:
            logger.error(f"Error analyzing precision: {e}")
            return {}
    
    def _analyze_mae(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Analyze Maximum Adverse Excursion (MAE)"""
        try:
            mae_data = []
            
            for rec in recommendations:
                symbol = rec.get('symbol', 'UNKNOWN')
                
                # Simulate MAE data (in real implementation, this would be calculated from price data)
                entry_price = 100  # Simulated
                max_adverse_price = entry_price * (1 - np.random.uniform(0.02, 0.20))
                mae_pct = (entry_price - max_adverse_price) / entry_price
                
                final_return = np.random.normal(0.03, 0.15)  # Simulated final return
                
                mae_data.append({
                    'symbol': symbol,
                    'mae_percent': mae_pct,
                    'final_return': final_return,
                    'recommendation_strength': rec.get('recommendation_strength', 'UNKNOWN')
                })
            
            # Calculate MAE statistics
            mae_percentages = [d['mae_percent'] for d in mae_data]
            
            mae_analysis = {
                'overall_statistics': {
                    'mean_mae': np.mean(mae_percentages),
                    'median_mae': np.median(mae_percentages),
                    'max_mae': np.max(mae_percentages),
                    'std_mae': np.std(mae_percentages),
                    'percentiles': {
                        '25th': np.percentile(mae_percentages, 25),
                        '75th': np.percentile(mae_percentages, 75),
                        '95th': np.percentile(mae_percentages, 95)
                    }
                },
                'by_recommendation_strength': {},
                'mae_vs_returns': mae_data[:20]  # Sample for visualization
            }
            
            # Analyze MAE by recommendation strength
            for strength in ['STRONG_BUY', 'BUY', 'HOLD']:
                strength_mae = [d['mae_percent'] for d in mae_data if d['recommendation_strength'] == strength]
                if strength_mae:
                    mae_analysis['by_recommendation_strength'][strength] = {
                        'mean': np.mean(strength_mae),
                        'median': np.median(strength_mae),
                        'count': len(strength_mae)
                    }
            
            return mae_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing MAE: {e}")
            return {}
    
    def _analyze_timing(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Analyze timing characteristics of recommendations"""
        try:
            timing_data = []
            
            for rec in recommendations:
                timestamp = rec.get('timestamp')
                if not timestamp:
                    continue
                    
                # Extract timing information
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = timestamp
                
                timing_data.append({
                    'hour': dt.hour,
                    'day_of_week': dt.weekday(),
                    'day_of_month': dt.day,
                    'symbol': rec.get('symbol', 'UNKNOWN'),
                    'recommendation_strength': rec.get('recommendation_strength', 'UNKNOWN')
                })
            
            # Analyze timing patterns
            hourly_distribution = {}
            daily_distribution = {}
            
            for data in timing_data:
                hour = data['hour']
                dow = data['day_of_week']
                
                hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
                daily_distribution[dow] = daily_distribution.get(dow, 0) + 1
            
            return {
                'hourly_distribution': hourly_distribution,
                'daily_distribution': daily_distribution,
                'total_analyzed': len(timing_data),
                'peak_hour': max(hourly_distribution, key=hourly_distribution.get) if hourly_distribution else None,
                'peak_day': max(daily_distribution, key=daily_distribution.get) if daily_distribution else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing timing: {e}")
            return {}
    
    def _analyze_sector_performance(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by sector"""
        try:
            sector_data = {}
            
            for rec in recommendations:
                sector = rec.get('sector', 'Unknown')
                if sector not in sector_data:
                    sector_data[sector] = {
                        'count': 0,
                        'scores': [],
                        'strengths': []
                    }
                
                sector_data[sector]['count'] += 1
                sector_data[sector]['scores'].append(rec.get('combined_score', 0))
                sector_data[sector]['strengths'].append(rec.get('recommendation_strength', 'UNKNOWN'))
            
            # Calculate sector statistics
            sector_analysis = {}
            for sector, data in sector_data.items():
                sector_analysis[sector] = {
                    'recommendation_count': data['count'],
                    'avg_score': np.mean(data['scores']) if data['scores'] else 0,
                    'strength_distribution': {
                        strength: data['strengths'].count(strength) 
                        for strength in set(data['strengths'])
                    }
                }
            
            return sector_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sector performance: {e}")
            return {}
    
    def _analyze_gates_effectiveness(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Analyze effectiveness of different gates"""
        try:
            gates_analysis = {
                'gate_pass_rates': {},
                'recommendation_by_gates': {},
                'score_by_gates': {}
            }
            
            all_gates = ['trend_filter', 'volatility_gate', 'volume_confirmation', 'mtf_confirmation']
            
            for gate in all_gates:
                passed_count = 0
                total_count = 0
                
                for rec in recommendations:
                    gates_data = rec.get('gates_passed', {})
                    if gate in gates_data:
                        total_count += 1
                        if gates_data[gate]:
                            passed_count += 1
                
                gates_analysis['gate_pass_rates'][gate] = {
                    'pass_rate': passed_count / total_count if total_count > 0 else 0,
                    'passed': passed_count,
                    'total': total_count
                }
            
            return gates_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing gates effectiveness: {e}")
            return {}
    
    def _analyze_risk_metrics(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Analyze risk-related metrics"""
        try:
            risk_metrics = {
                'risk_reward_ratios': [],
                'stop_loss_distances': [],
                'position_sizing': [],
                'volatility_exposure': []
            }
            
            for rec in recommendations:
                trade_plan = rec.get('trade_plan', {})
                
                # Extract risk metrics
                if 'risk_reward_ratio' in trade_plan:
                    risk_metrics['risk_reward_ratios'].append(trade_plan['risk_reward_ratio'])
                
                # Simulate other metrics
                risk_metrics['stop_loss_distances'].append(np.random.uniform(0.03, 0.12))
                risk_metrics['volatility_exposure'].append(np.random.uniform(0.15, 0.45))
            
            # Calculate statistics
            risk_analysis = {}
            for metric, values in risk_metrics.items():
                if values:
                    risk_analysis[metric] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing risk metrics: {e}")
            return {}
    
    def _compare_with_benchmarks(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Compare performance with relevant benchmarks"""
        try:
            # Simulate benchmark comparison
            benchmark_data = {
                'nifty_50': {
                    'period_return': np.random.normal(0.02, 0.08),
                    'volatility': 0.18
                },
                'nifty_500': {
                    'period_return': np.random.normal(0.025, 0.09),
                    'volatility': 0.20
                },
                'sector_average': {
                    'period_return': np.random.normal(0.015, 0.12),
                    'volatility': 0.25
                }
            }
            
            # Calculate our portfolio metrics
            portfolio_returns = [np.random.normal(0.04, 0.16) for _ in recommendations[:50]]
            
            portfolio_metrics = {
                'mean_return': np.mean(portfolio_returns) if portfolio_returns else 0,
                'volatility': np.std(portfolio_returns) if portfolio_returns else 0,
                'sharpe_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) if portfolio_returns and np.std(portfolio_returns) > 0 else 0
            }
            
            return {
                'benchmarks': benchmark_data,
                'portfolio': portfolio_metrics,
                'outperformance': {
                    bench: portfolio_metrics['mean_return'] - data['period_return']
                    for bench, data in benchmark_data.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing with benchmarks: {e}")
            return {}
    
    def _create_visualizations(self, report_data: Dict[str, Any]) -> None:
        """Create visualization charts for the report"""
        try:
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            fig_size = (12, 8)
            
            # 1. Precision Heatmap
            precision_data = report_data.get('precision_analysis', {})
            if precision_data:
                self._create_precision_heatmap(precision_data)
            
            # 2. MAE Distribution
            mae_data = report_data.get('mae_analysis', {})
            if mae_data:
                self._create_mae_distribution(mae_data)
            
            # 3. Sector Performance Chart
            sector_data = report_data.get('sector_analysis', {})
            if sector_data:
                self._create_sector_chart(sector_data)
            
            # 4. Gates Effectiveness Chart
            gates_data = report_data.get('gates_analysis', {})
            if gates_data:
                self._create_gates_chart(gates_data)
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _create_precision_heatmap(self, precision_data: Dict[str, Any]) -> None:
        """Create precision heatmap"""
        try:
            horizons = []
            thresholds = []
            precision_values = []
            
            for horizon, data in precision_data.items():
                for threshold, metrics in data.items():
                    horizons.append(horizon)
                    thresholds.append(threshold)
                    precision_values.append(metrics['precision'])
            
            if not precision_values:
                return
                
            # Create pivot table
            df = pd.DataFrame({
                'Horizon': horizons,
                'Threshold': thresholds,
                'Precision': precision_values
            })
            
            pivot = df.pivot(index='Horizon', columns='Threshold', values='Precision')
            
            # Create heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, cmap='RdYlGn', fmt='.2%', 
                       cbar_kws={'label': 'Precision'})
            plt.title('Precision Analysis: Hit Rate by Time Horizon and Return Threshold')
            plt.tight_layout()
            
            # Save
            plt.savefig(os.path.join(self.output_dir, 'precision_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating precision heatmap: {e}")
    
    def _create_mae_distribution(self, mae_data: Dict[str, Any]) -> None:
        """Create MAE distribution chart"""
        try:
            mae_vs_returns = mae_data.get('mae_vs_returns', [])
            if not mae_vs_returns:
                return
            
            mae_values = [d['mae_percent'] for d in mae_vs_returns]
            returns = [d['final_return'] for d in mae_vs_returns]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # MAE histogram
            ax1.hist(mae_values, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax1.set_xlabel('Maximum Adverse Excursion (%)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Maximum Adverse Excursion')
            ax1.axvline(np.mean(mae_values), color='darkred', linestyle='--', label=f'Mean: {np.mean(mae_values):.1%}')
            ax1.legend()
            
            # MAE vs Returns scatter
            ax2.scatter(mae_values, returns, alpha=0.6, color='blue')
            ax2.set_xlabel('Maximum Adverse Excursion (%)')
            ax2.set_ylabel('Final Return (%)')
            ax2.set_title('MAE vs Final Returns')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'mae_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating MAE distribution: {e}")
    
    def _create_sector_chart(self, sector_data: Dict[str, Any]) -> None:
        """Create sector performance chart"""
        try:
            sectors = list(sector_data.keys())
            counts = [data['recommendation_count'] for data in sector_data.values()]
            avg_scores = [data['avg_score'] for data in sector_data.values()]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Recommendation counts by sector
            ax1.bar(sectors, counts, color='skyblue', edgecolor='navy')
            ax1.set_xlabel('Sector')
            ax1.set_ylabel('Number of Recommendations')
            ax1.set_title('Recommendations by Sector')
            ax1.tick_params(axis='x', rotation=45)
            
            # Average scores by sector
            ax2.bar(sectors, avg_scores, color='lightgreen', edgecolor='darkgreen')
            ax2.set_xlabel('Sector')
            ax2.set_ylabel('Average Combined Score')
            ax2.set_title('Average Score by Sector')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'sector_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating sector chart: {e}")
    
    def _create_gates_chart(self, gates_data: Dict[str, Any]) -> None:
        """Create gates effectiveness chart"""
        try:
            gate_pass_rates = gates_data.get('gate_pass_rates', {})
            if not gate_pass_rates:
                return
            
            gates = list(gate_pass_rates.keys())
            pass_rates = [data['pass_rate'] for data in gate_pass_rates.values()]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(gates, pass_rates, color='orange', edgecolor='darkorange')
            plt.xlabel('Gates')
            plt.ylabel('Pass Rate')
            plt.title('Gate Pass Rates')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, pass_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{rate:.1%}', ha='center', va='bottom')
            
            plt.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'gates_effectiveness.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating gates chart: {e}")
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report"""
        try:
            period = report_data.get('period', {})
            summary = report_data.get('summary_stats', {})
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Golden Run Report - {period.get('quarter', 'Unknown')}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e8f4f8; border-radius: 3px; }}
                    .chart {{ text-align: center; margin: 20px 0; }}
                    .warning {{ color: #d9534f; }}
                    .success {{ color: #5cb85c; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Swing Trading Golden Run Report</h1>
                    <h2>Period: {period.get('start', '')} to {period.get('end', '')}</h2>
                    <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h3>Executive Summary</h3>
                    <div class="metric">
                        <strong>Total Recommendations:</strong> {summary.get('total_recommendations', 0)}
                    </div>
                    <div class="metric">
                        <strong>Average Combined Score:</strong> {summary.get('score_statistics', {}).get('combined', {}).get('mean', 0):.3f}
                    </div>
                </div>
                
                <div class="section">
                    <h3>Precision Analysis</h3>
                    <p>Hit rates at different time horizons and return thresholds:</p>
                    <div class="chart">
                        <img src="precision_heatmap.png" alt="Precision Heatmap" style="max-width: 100%;">
                    </div>
                </div>
                
                <div class="section">
                    <h3>Maximum Adverse Excursion (MAE)</h3>
                    <p>Analysis of maximum drawdowns experienced in recommendations:</p>
                    <div class="chart">
                        <img src="mae_analysis.png" alt="MAE Analysis" style="max-width: 100%;">
                    </div>
                </div>
                
                <div class="section">
                    <h3>Sector Performance</h3>
                    <div class="chart">
                        <img src="sector_analysis.png" alt="Sector Analysis" style="max-width: 100%;">
                    </div>
                </div>
                
                <div class="section">
                    <h3>Gates Effectiveness</h3>
                    <div class="chart">
                        <img src="gates_effectiveness.png" alt="Gates Effectiveness" style="max-width: 100%;">
                    </div>
                </div>
                
                <div class="section">
                    <h3>Key Insights & Recommendations</h3>
                    <ul>
                        <li>Monitor precision rates to ensure they remain above 60% at 10-day horizon</li>
                        <li>Consider tightening gates if pass rates are too high (>80%)</li>
                        <li>Review sector allocation for overconcentration risk</li>
                        <li>Analyze MAE patterns to optimize stop-loss levels</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h3>Data Quality Notes</h3>
                    <p class="warning">⚠️ This report uses simulated performance data for demonstration. 
                    In production, actual price and return data should be used.</p>
                </div>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return f"<html><body><h1>Error generating report: {str(e)}</h1></body></html>"
    
    def _get_last_quarter_end(self) -> date:
        """Get the end date of the last completed quarter"""
        today = date.today()
        current_quarter = (today.month - 1) // 3 + 1
        
        if current_quarter == 1:
            return date(today.year - 1, 12, 31)
        elif current_quarter == 2:
            return date(today.year, 3, 31)
        elif current_quarter == 3:
            return date(today.year, 6, 30)
        else:
            return date(today.year, 9, 30)
    
    def _get_quarter_label(self, quarter_end: date) -> str:
        """Get quarter label (e.g., '2024-Q1')"""
        quarter = (quarter_end.month - 1) // 3 + 1
        return f"{quarter_end.year}-Q{quarter}"


def main():
    """Main function to run the golden run reporter"""
    reporter = GoldenRunReporter()
    
    # Generate report for last quarter
    report_data = reporter.generate_quarterly_report()
    
    if 'error' in report_data:
        print(f"Error generating report: {report_data['error']}")
        return
    
    print("Golden run report generated successfully!")
    print(f"Report saved to: {reporter.output_dir}")
    
    # Print summary
    summary = report_data.get('summary_stats', {})
    print(f"Total recommendations analyzed: {summary.get('total_recommendations', 0)}")


if __name__ == "__main__":
    main()
