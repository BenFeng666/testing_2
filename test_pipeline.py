"""
Test script for Active Learning Pipeline
Test individual components before running the full pipeline
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import yaml
        print("  ✓ yaml")
    except ImportError:
        print("  ✗ yaml - Install with: pip3 install pyyaml")
        return False
    
    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - Install with: pip3 install pandas")
        return False
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - Install with: pip3 install numpy")
        return False
    
    try:
        import scipy
        print("  ✓ scipy")
    except ImportError:
        print("  ✗ scipy - Install with: pip3 install scipy")
        return False
    
    try:
        from confidence_calculator import ConfidenceCalculator
        print("  ✓ confidence_calculator")
    except ImportError as e:
        print(f"  ✗ confidence_calculator - {e}")
        return False
    
    try:
        from human_feedback import HumanFeedbackInterface
        print("  ✓ human_feedback")
    except ImportError as e:
        print(f"  ✗ human_feedback - {e}")
        return False
    
    print("\n✓ All basic imports successful!\n")
    return True

def test_config():
    """Test configuration file"""
    print("Testing configuration file...")
    
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("  ✓ Config file loaded successfully")
        print(f"  - Confidence threshold: {config['thresholds']['confidence_threshold']}")
        print(f"  - Score threshold: {config['thresholds']['target_score_threshold']}")
        print(f"  - Pairs per round: {config['human_feedback']['num_pairs_per_round']}")
        print(f"  - Max rounds: {config['human_feedback']['max_rounds']}")
        return True
    except Exception as e:
        print(f"  ✗ Config file error: {e}")
        return False

def test_confidence_calculator():
    """Test confidence calculator"""
    print("\nTesting confidence calculator...")
    
    try:
        from confidence_calculator import ConfidenceCalculator
        
        calc = ConfidenceCalculator(score_min=1, score_max=10)
        
        # Test high confidence
        high_conf = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        result1 = calc.calculate_confidence_from_samples(high_conf)
        print(f"  High confidence test:")
        print(f"    - Score: {result1['mean_score']:.2f}")
        print(f"    - Confidence: {result1['confidence']:.2f}")
        assert result1['confidence'] > 9.0, "High confidence should be > 9.0"
        
        low_range = [0.12, 0.25, 0.55, 0.61, 0.49, 0.7, 0.9, 0.3, 0.8, 0.4]
        result1 = calc.calculate_confidence_from_samples(low_range)
        print(f"  low_range test:")
        print(f"    - Score: {result1['mean_score']:.2f}")
        print(f"    - Confidence: {result1['confidence']:.2f}")

        identical = [0.5] * 10
        result1 = calc.calculate_confidence_from_samples(identical)
        print(f"  identical test:")
        print(f"    - Score: {result1['mean_score']:.2f}")
        print(f"    - Confidence: {result1['confidence']:.2f}")
        
        
        # Test low confidence
        low_conf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result2 = calc.calculate_confidence_from_samples(low_conf)
        print(f"  Low confidence test:")
        print(f"    - Score: {result2['mean_score']:.2f}")
        print(f"    - Confidence: {result2['confidence']:.2f}")
        assert result2['confidence'] < 2.0, "Low confidence should be < 2.0"
        
        print("  ✓ Confidence calculator working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Confidence calculator error: {e}")
        return False

def test_data_loading():
    """Test loading test dataset"""
    print("\nTesting data loading...")
    
    try:
        import pandas as pd
        
        # Try to load the test set
        test_file = "dataset/1600set.xlsx"
        
        if not os.path.exists(test_file):
            print(f"  ⚠ Test file not found: {test_file}")
            print(f"    Make sure the file exists before running the pipeline")
            return True  # Not a critical error for testing
        
        # Try different ways to read
        try:
            df = pd.read_excel(test_file, header=1)
        except:
            df = pd.read_excel(test_file)
        
        print(f"  ✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  - Columns: {df.columns.tolist()[:5]}...")  # Show first 5 columns
        
        return True
    except Exception as e:
        print(f"  ✗ Data loading error: {e}")
        return False

def test_human_feedback():
    """Test human feedback interface"""
    print("\nTesting human feedback interface...")
    
    try:
        from human_feedback import HumanFeedbackInterface
        
        # Create test interface
        interface = HumanFeedbackInterface("test_feedback.json")
        
        # Create test molecules
        test_mols = [
            {'smiles': 'CCO', 'predicted_score': 6.5, 'confidence': 7.2, 'entropy': 0.8},
            {'smiles': 'CCCO', 'predicted_score': 7.1, 'confidence': 6.8, 'entropy': 0.9},
            {'smiles': 'CCCCO', 'predicted_score': 5.5, 'confidence': 6.5, 'entropy': 1.0},
            {'smiles': 'CCCCCO', 'predicted_score': 8.2, 'confidence': 7.0, 'entropy': 0.85},
        ]
        
        # Test pair selection
        pairs = interface.select_pairs(test_mols, num_pairs=2, strategy='entropy')
        print(f"  ✓ Pair selection works: {len(pairs)} pairs selected")
        
        # Clean up test file
        if os.path.exists("test_feedback.json"):
            os.remove("test_feedback.json")
        
        return True
    except Exception as e:
        print(f"  ✗ Human feedback error: {e}")
        return False

def test_output_directory():
    """Test output directory creation"""
    print("\nTesting output directory...")
    
    try:
        os.makedirs("output", exist_ok=True)
        print("  ✓ Output directory ready")
        return True
    except Exception as e:
        print(f"  ✗ Output directory error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("ACTIVE LEARNING PIPELINE - TEST SUITE")
    print("="*80)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Confidence Calculator", test_confidence_calculator),
        ("Data Loading", test_data_loading),
        ("Human Feedback", test_human_feedback),
        ("Output Directory", test_output_directory),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! You can now run the pipeline:")
        print("  python3 active_learning_pipeline.py")
    else:
        print("\n⚠ Some tests failed. Please fix the issues before running the pipeline.")
        print("  Check the error messages above for details.")
    
    print("="*80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

