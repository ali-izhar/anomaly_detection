#!/usr/bin/env python

"""
Test script to verify run_experiments.py functionality
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def test_imports():
    """Test that all required imports work"""
    try:
        from src.run_experiments import (
            setup_logging, 
            save_results_robust, 
            standardize_metrics,
            get_base_config
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_excel_support():
    """Test that Excel writing is supported"""
    try:
        import pandas as pd
        import openpyxl
        
        # Create test DataFrame
        test_data = {
            'name': ['test1', 'test2'],
            'TPR': [0.85, 0.92],
            'FPR': [0.1, 0.05]
        }
        df = pd.DataFrame(test_data)
        
        # Test Excel writing
        test_path = "test_output.xlsx"
        with pd.ExcelWriter(test_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Test', index=False)
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
            
        print("✓ Excel support working")
        return True
    except Exception as e:
        print(f"✗ Excel support error: {e}")
        return False

def test_standardize_metrics():
    """Test the standardize_metrics function"""
    try:
        from src.run_experiments import standardize_metrics
        
        # Test metrics
        test_metrics = {
            'name': 'test_exp',
            'TPR': 0.85,
            'network': 'sbm',
            'custom_field': 'custom_value'
        }
        
        standardized = standardize_metrics(test_metrics, 'test')
        
        # Check required fields are present
        required_fields = ['name', 'experiment_type', 'network', 'TPR', 'FPR', 'ADD']
        for field in required_fields:
            if field not in standardized:
                print(f"✗ Missing required field: {field}")
                return False
        
        # Check custom field is preserved
        if standardized.get('custom_field') != 'custom_value':
            print("✗ Custom field not preserved")
            return False
            
        print("✓ Metrics standardization working")
        return True
    except Exception as e:
        print(f"✗ Metrics standardization error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing run_experiments.py improvements...")
    print("-" * 50)
    
    tests = [
        test_imports,
        test_excel_support,
        test_standardize_metrics
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("-" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The improvements should work correctly.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 