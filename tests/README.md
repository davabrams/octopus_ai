# Octopus AI Test Suite

This directory contains comprehensive unit tests for the octopus_ai project, covering all major components of the simulation, training, and inference systems.

## Test Structure

### Core Test Files

1. **`test_simulator.py`** - *Original tests (expanded)*
   - `TestSurfaceGenerator` - Random surface generation
   - `TestAgentGenerator` - Agent creation and movement
   - `TestKinematicPrimitives` - Basic State class functionality

2. **`test_octopus_generator.py`** - *NEW - Core simulation components*
   - `TestSucker` - Individual sucker behavior and color inference
   - `TestLimb` - Limb management, sucker coordination, and movement
   - `TestOctopus` - Full octopus simulation with parallel processing
   - `TestIntegration` - Complete simulation workflows

3. **`test_utilities.py`** - *NEW - Utility functions*
   - `TestTrainTestSplit` - Data splitting functions
   - `TestOctoNorm` - Color normalization ([0,1] ↔ [-1,1])
   - `TestTensorFlowConversion` - Dataset conversion utilities
   - `TestLogErasure` - Log cleanup functionality
   - `TestDefaultLoader` - Abstract loader base class

4. **`test_training_losses.py`** - *NEW - Custom loss functions*
   - `TestConstraintLoss` - Constraint-based loss function
   - `TestWeightedSumLoss` - Weighted combination of multiple losses
   - `TestLossIntegration` - Loss functions in Keras models

5. **`test_inference_server.py`** - *NEW - Server components*
   - `TestInferenceJob` - Job creation, execution, and status tracking
   - `TestInferenceQueue` - Job queue management and threading
   - `TestExecuteSuckerInference` - Model inference execution
   - `TestInferenceServerIntegration` - Complete server workflows

6. **`test_kinematics.py`** - *NEW - Physics and optimization*
   - `TestState` - Enhanced kinematic state management
   - `TestAgent` - Agent-specific behavior and inheritance
   - `TestColor` - Color data structure
   - `TestCostFunction` classes - ILQR cost functions and optimization

7. **`test_trainers.py`** - *NEW - Training pipelines*
   - `TestTrainer` - Base trainer class
   - `TestSuckerTrainer` - Sucker model training pipeline
   - `TestLimbTrainer` - Limb model training with ragged tensors
   - `TestTrainingIntegration` - End-to-end training workflows

8. **`test_integration.py`** - *NEW - End-to-end system tests*
   - `TestSimulationIntegration` - Complete simulation cycles
   - `TestTrainingIntegration` - Training pipeline integration
   - `TestInferenceServerIntegration` - Server integration
   - `TestDataGenerationIntegration` - Data generation workflows
   - `TestEndToEndWorkflow` - Complete system workflows

## Test Coverage

### High Priority Components (Fully Tested)
- ✅ **Core Simulation**: Sucker, Limb, Octopus classes
- ✅ **Training System**: Loss functions, trainers, data processing
- ✅ **Inference Server**: Job management, queue operations, model inference
- ✅ **Utility Functions**: Data processing, normalization, file operations
- ✅ **Kinematics & Physics**: State management, ILQR cost functions

### Medium Priority Components (Tested)
- ✅ **Data Generation**: Synthetic data creation workflows
- ✅ **Configuration Management**: Parameter validation and consistency
- ✅ **Model Serialization**: Save/load functionality

### Integration Testing
- ✅ **End-to-End Workflows**: Complete simulation → training → inference
- ✅ **Multi-Component Interaction**: Agent-octopus interactions
- ✅ **Error Handling**: Edge cases and failure scenarios

## Test Statistics

- **Total Test Files**: 8
- **Test Classes**: ~25
- **Individual Tests**: ~150+
- **Lines of Test Code**: ~2,500+

## Key Testing Features

### Mocking and Isolation
- Extensive use of `unittest.mock` to isolate components
- TensorFlow model mocking for training tests
- File system mocking for I/O operations
- Threading mocks for inference server tests

### Property-Based Testing
- Mathematical function validation (distances, gradients)
- Constraint verification (parameter ranges, data shapes)
- Behavioral consistency checks

### Integration Scenarios
- Complete simulation cycles with all components
- Training pipeline from data generation to model deployment
- Inference server job lifecycle management
- Multi-agent interaction scenarios

### Error Handling
- Invalid input validation
- File system error scenarios
- Model training failure handling
- Network and threading error conditions

## Running Tests

### 🚀 **One Command to Run All Tests**

**Easiest Methods:**
```bash
# Method 1: Using the test runner script
python run_tests.py

# Method 2: Using Make (if available)
make test

# Method 3: Using shell script
./test.sh
```

### **Advanced Test Options**

**Using the test runner script:**
```bash
# Verbose output
python run_tests.py --verbose

# With coverage report
python run_tests.py --coverage

# Using unittest instead of pytest
python run_tests.py --runner unittest

# Using Bazel
python run_tests.py --runner bazel

# Run specific test file
python run_tests.py --test test_octopus_generator.py

# Check dependencies only
python run_tests.py --check-deps
```

**Using Make commands:**
```bash
make test              # Run all tests
make test-verbose      # Verbose output
make test-coverage     # With coverage
make test-unittest     # Using unittest
make test-bazel        # Using Bazel
make test-octopus      # Just octopus tests
make test-utilities    # Just utility tests
make install-deps      # Install test dependencies
make clean             # Clean test artifacts
```

### **Traditional Methods**

**Direct pytest:**
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test file
python -m pytest tests/test_octopus_generator.py -v

# Run specific test class
python -m pytest tests/test_utilities.py::TestTrainTestSplit -v

# Run specific test method
python -m pytest tests/test_training_losses.py::TestConstraintLoss::test_constraint_loss_call_perfect_match -v
```

**Direct unittest:**
```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py" -v

# Run specific test file
python -m unittest tests.test_octopus_generator -v
```

**Using Bazel:**
```bash
# Run all tests
bazel test //tests:all

# Run specific test target
bazel test //tests:test_octopus_generator
bazel test //tests:test_utilities
bazel test //tests:test_training_losses
bazel test //tests:test_inference_server
bazel test //tests:test_kinematics
bazel test //tests:test_trainers
bazel test //tests:test_integration
```

## Test Design Principles

1. **Isolation**: Each test is independent and doesn't rely on external state
2. **Determinism**: Tests use fixed random seeds where applicable
3. **Comprehensiveness**: Both happy paths and edge cases are tested
4. **Performance**: Integration tests use minimal parameters for speed
5. **Maintainability**: Clear test names and extensive documentation

## Mock Strategy

### External Dependencies
- **TensorFlow Models**: Mocked to return predictable outputs
- **File System**: Temporary files and mocked I/O operations
- **Threading**: Controlled execution for deterministic testing
- **Random Operations**: Seeded for reproducible results

### Internal Components
- **Surface Generation**: Mocked for consistent test environments
- **Agent Behavior**: Controlled for predictable interactions
- **Time Operations**: Fixed timestamps for consistent testing

## Future Enhancements

1. **Performance Testing**: Benchmarking for large-scale simulations
2. **Stress Testing**: High-load scenarios for inference server
3. **Visual Testing**: Validation of visualization components
4. **Hardware Testing**: GPU/CPU performance comparisons
5. **Network Testing**: Distributed simulation scenarios

## Contributing

When adding new functionality:

1. **Write tests first** (TDD approach)
2. **Mock external dependencies** appropriately
3. **Test both success and failure scenarios**
4. **Update this README** with new test descriptions
5. **Ensure all tests pass** before submitting changes

## Test Environment

- **Python**: 3.11+
- **TensorFlow**: 2.x
- **Testing Framework**: unittest + pytest
- **Mocking**: unittest.mock
- **Build System**: Bazel (with py_test targets)

This comprehensive test suite ensures the reliability, maintainability, and correctness of the octopus_ai simulation system across all its components and use cases.