"""
Run Analysis Module
===================

This module orchestrates the automated stock analysis
for scheduling or manual execution.
Extracted from run_analysis.py for modularity.
"""

import sys
import os

# Add parent directories to path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from core.analysis.analysis_orchestrator import AnalysisOrchestrator
from utils.logger import setup_logging
from datetime import datetime

logger = setup_logging()

class RunAnalysis:
    """
    Orchestrates the stock analysis workflow.
    """

    def __init__(self):
        self.orchestrator = AnalysisOrchestrator()

    def execute(self):
        """
        Execute the analysis.
        """
        try:
            logger.info("Starting RunAnalysis execution.")
            # Logic to orchestrate an analysis event
        except Exception as e:
            logger.error(f"Error executing RunAnalysis: {e}")
