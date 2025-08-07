#!/usr/bin/env python3
"""
Comprehensive test runner for octopus_ai project
Run all unit tests with: python run_tests.py
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_pytest_tests(verbose=False, coverage=False, specific_test=None):
    """Run tests using pytest"""
    print("🧪 Running octopus_ai test suite with pytest...")
    
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing", "--cov-report=html"])
    
    if specific_test:
        cmd = [sys.executable, "-m", "pytest", f"tests/{specific_test}", "-v"]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=False)
        return result.returncode == 0
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest pytest-cov")
        return False

def run_unittest_tests(verbose=False):
    """Run tests using unittest discover"""
    print("🧪 Running octopus_ai test suite with unittest...")
    
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running unittest: {e}")
        return False

def run_bazel_tests():
    """Run tests using Bazel"""
    print("🧪 Running octopus_ai test suite with Bazel...")
    
    try:
        result = subprocess.run(["bazel", "test", "//tests:all"], 
                              cwd=Path(__file__).parent, capture_output=False)
        return result.returncode == 0
    except FileNotFoundError:
        print("❌ Bazel not found. Skipping Bazel tests.")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking test dependencies...")
    
    missing_deps = []
    
    try:
        import numpy
        print("✅ numpy available")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import tensorflow
        print("✅ tensorflow available")
    except ImportError:
        missing_deps.append("tensorflow")
    
    try:
        import pytest
        print("✅ pytest available")
    except ImportError:
        print("⚠️  pytest not available (will use unittest)")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install numpy tensorflow pytest pytest-cov")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run octopus_ai test suite")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Verbose output")
    parser.add_argument("-c", "--coverage", action="store_true",
                       help="Run with coverage report (pytest only)")
    parser.add_argument("-r", "--runner", choices=["pytest", "unittest", "bazel"],
                       default="pytest", help="Test runner to use")
    parser.add_argument("-t", "--test", type=str,
                       help="Run specific test file (e.g., test_utilities.py)")
    parser.add_argument("--check-deps", action="store_true",
                       help="Only check dependencies")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🐙 OCTOPUS AI TEST SUITE")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Check dependencies
    if not check_dependencies():
        if not args.check_deps:
            print("\n❌ Cannot run tests due to missing dependencies")
            return 1
        else:
            return 1
    
    if args.check_deps:
        print("\n✅ All dependencies available")
        return 0
    
    print(f"\n🎯 Running tests with {args.runner}...")
    
    success = False
    
    if args.runner == "pytest":
        success = run_pytest_tests(args.verbose, args.coverage, args.test)
    elif args.runner == "unittest":
        if args.test or args.coverage:
            print("⚠️  --test and --coverage options only available with pytest")
        success = run_unittest_tests(args.verbose)
    elif args.runner == "bazel":
        if args.test or args.coverage:
            print("⚠️  --test and --coverage options only available with pytest")
        success = run_bazel_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("🐙 Octopus AI test suite completed successfully")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🐙 Check output above for details")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())