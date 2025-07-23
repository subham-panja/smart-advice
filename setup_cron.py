#!/usr/bin/env python3
"""
Cron Job Setup Script
File: setup_cron.py

This script sets up a cron job to run the automated stock analysis every hour.
"""

import os
import subprocess
import sys
from datetime import datetime

def get_project_path():
    """Get the absolute path to the project directory."""
    return os.path.dirname(os.path.abspath(__file__))

def get_python_path():
    """Get the path to the Python interpreter in the virtual environment."""
    project_path = get_project_path()
    venv_python = os.path.join(project_path, 'venv', 'bin', 'python')
    
    if os.path.exists(venv_python):
        return venv_python
    else:
        print(f"Warning: Virtual environment Python not found at {venv_python}")
        return sys.executable

def create_cron_entry():
    """Create the cron job entry."""
    project_path = get_project_path()
    python_path = get_python_path()
    script_path = os.path.join(project_path, 'run_analysis.py')
    log_path = os.path.join(project_path, 'logs', 'cron_analysis.log')
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Create cron entry (runs every hour)
    cron_entry = f"0 * * * * {python_path} {script_path} >> {log_path} 2>&1"
    
    return cron_entry

def install_cron_job():
    """Install the cron job."""
    try:
        # Get current crontab
        try:
            current_crontab = subprocess.check_output(['crontab', '-l'], stderr=subprocess.STDOUT).decode('utf-8')
        except subprocess.CalledProcessError:
            current_crontab = ""
        
        # Create new cron entry
        new_entry = create_cron_entry()
        
        # Check if entry already exists
        if 'run_analysis.py' in current_crontab:
            print("Cron job for stock analysis already exists.")
            print("Current entry found in crontab.")
            return True
        
        # Add new entry
        updated_crontab = current_crontab + '\\n' + new_entry + '\\n'
        
        # Install updated crontab
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(updated_crontab.encode('utf-8'))
        
        if process.returncode == 0:
            print("✓ Cron job installed successfully!")
            print(f"Analysis will run every hour with logs at: {os.path.join(get_project_path(), 'logs', 'cron_analysis.log')}")
            return True
        else:
            print(f"✗ Failed to install cron job: {stderr.decode('utf-8')}")
            return False
            
    except Exception as e:
        print(f"✗ Error installing cron job: {e}")
        return False

def uninstall_cron_job():
    """Remove the cron job."""
    try:
        # Get current crontab
        try:
            current_crontab = subprocess.check_output(['crontab', '-l'], stderr=subprocess.STDOUT).decode('utf-8')
        except subprocess.CalledProcessError:
            print("No crontab found.")
            return True
        
        # Remove lines containing run_analysis.py
        lines = current_crontab.split('\\n')
        updated_lines = [line for line in lines if 'run_analysis.py' not in line]
        updated_crontab = '\\n'.join(updated_lines)
        
        # Install updated crontab
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(updated_crontab.encode('utf-8'))
        
        if process.returncode == 0:
            print("✓ Cron job removed successfully!")
            return True
        else:
            print(f"✗ Failed to remove cron job: {stderr.decode('utf-8')}")
            return False
            
    except Exception as e:
        print(f"✗ Error removing cron job: {e}")
        return False

def show_cron_status():
    """Show current cron job status."""
    try:
        current_crontab = subprocess.check_output(['crontab', '-l'], stderr=subprocess.STDOUT).decode('utf-8')
        
        # Find lines containing run_analysis.py
        relevant_lines = [line for line in current_crontab.split('\\n') if 'run_analysis.py' in line]
        
        if relevant_lines:
            print("Current stock analysis cron jobs:")
            for line in relevant_lines:
                print(f"  {line}")
        else:
            print("No stock analysis cron jobs found.")
            
    except subprocess.CalledProcessError:
        print("No crontab found.")
    except Exception as e:
        print(f"Error checking cron status: {e}")

def test_analysis_script():
    """Test the analysis script with a small number of stocks."""
    print("Testing analysis script...")
    
    project_path = get_project_path()
    python_path = get_python_path()
    script_path = os.path.join(project_path, 'run_analysis.py')
    
    try:
        # Run with test flag
        result = subprocess.run([python_path, script_path, '--test'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Analysis script test completed successfully!")
            print("Last few lines of output:")
            output_lines = result.stdout.strip().split('\\n')
            for line in output_lines[-5:]:
                print(f"  {line}")
        else:
            print("✗ Analysis script test failed!")
            print("Error output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("✗ Analysis script test timed out after 5 minutes")
    except Exception as e:
        print(f"✗ Error testing analysis script: {e}")

def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup cron job for automated stock analysis')
    parser.add_argument('action', choices=['install', 'uninstall', 'status', 'test'], 
                       help='Action to perform')
    
    args = parser.parse_args()
    
    print("=== Stock Analysis Cron Job Manager ===")
    print(f"Project path: {get_project_path()}")
    print(f"Python path: {get_python_path()}")
    print(f"Action: {args.action}")
    print()
    
    if args.action == 'install':
        print("Installing cron job to run stock analysis every hour...")
        if install_cron_job():
            print("\\nCron job details:")
            print(f"Schedule: Every hour (0 * * * *)")
            print(f"Script: {os.path.join(get_project_path(), 'run_analysis.py')}")
            print(f"Logs: {os.path.join(get_project_path(), 'logs', 'cron_analysis.log')}")
            print("\\nTo check if it's working, wait for the next hour and check the logs.")
        
    elif args.action == 'uninstall':
        print("Removing cron job...")
        uninstall_cron_job()
        
    elif args.action == 'status':
        print("Checking cron job status...")
        show_cron_status()
        
    elif args.action == 'test':
        print("Testing analysis script...")
        test_analysis_script()

if __name__ == "__main__":
    main()
