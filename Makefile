# Octopus AI Project Makefile

.PHONY: test test-verbose test-coverage test-unittest test-bazel help install-deps clean

# Default target
help:
	@echo "🐙 Octopus AI Test Commands"
	@echo "=========================="
	@echo "make test          - Run all tests (default: pytest)"
	@echo "make test-verbose  - Run tests with verbose output"
	@echo "make test-coverage - Run tests with coverage report"
	@echo "make test-unittest - Run tests with unittest"
	@echo "make test-bazel    - Run tests with Bazel"
	@echo "make install-deps  - Install test dependencies"
	@echo "make clean         - Clean test artifacts"
	@echo "make help          - Show this help message"

# Main test targets
test:
	@python run_tests.py

test-verbose:
	@python run_tests.py --verbose

test-coverage:
	@python run_tests.py --coverage

test-unittest:
	@python run_tests.py --runner unittest

test-bazel:
	@python run_tests.py --runner bazel

# Specific test file targets
test-octopus:
	@python run_tests.py --test test_octopus_generator.py

test-utilities:
	@python run_tests.py --test test_utilities.py

test-training:
	@python run_tests.py --test test_training_losses.py

test-inference:
	@python run_tests.py --test test_inference_server.py

test-kinematics:
	@python run_tests.py --test test_kinematics.py

test-trainers:
	@python run_tests.py --test test_trainers.py

test-integration:
	@python run_tests.py --test test_integration.py

# Dependency management
install-deps:
	@echo "Installing test dependencies..."
	@pip install pytest pytest-cov numpy tensorflow

check-deps:
	@python run_tests.py --check-deps

# Cleanup
clean:
	@echo "Cleaning test artifacts..."
	@rm -rf .pytest_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf __pycache__/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned test artifacts"

# Quick lint check
lint:
	@echo "Running flake8 linter..."
	@flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || echo "⚠️  Install flake8 with: pip install flake8"

# Development helpers
dev-setup: install-deps
	@echo "Setting up development environment..."
	@pip install flake8 black isort
	@echo "✅ Development environment ready"

# CI/CD target
ci-test: clean test-coverage
	@echo "✅ CI test pipeline completed"