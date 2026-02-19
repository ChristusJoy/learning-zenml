# Learning ZenML - Step by Step

A hands-on tutorial repository to learn ZenML from scratch. This repository contains progressively complex examples that teach you the core concepts of ZenML pipelines.

## About

ZenML is an extensible, open-source MLOps framework for creating portable, production-ready ML pipelines. This repository provides a structured learning path with 6 practical examples that build on each other. 
After completing this tutorial feel free to explore an end to end MLOps project using ZenML with data drift simulation [here](https://github.com/ChristusJoy/vehicle-insurance-zenml).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Learning Path](#learning-path)
   - [Step 1: Hello Pipeline](#step-1-hello-pipeline---your-first-zenml-pipeline)
   - [Step 2: I/O Pipeline](#step-2-io-pipeline---working-with-inputs-and-outputs)
   - [Step 3: Parameterized Pipeline](#step-3-parameterized-pipeline---configurable-pipelines)
   - [Step 4: Cached Pipeline](#step-4-cached-pipeline---optimizing-with-caching)
   - [Step 5: Metadata Pipeline](#step-5-metadata-pipeline---logging-metrics-and-metadata)
   - [Step 6: Tagged Pipeline](#step-6-tagged-pipeline---organizing-artifacts-with-tags)
4. [ZenML Dashboard](#zenml-dashboard)
5. [Key Concepts Summary](#key-concepts-summary)
6. [Next Steps](#next-steps)

---

## Prerequisites

Before starting, make sure you have:

- Python 3.8 or higher installed
- Basic understanding of Python decorators
- Familiarity with ML/data science concepts (helpful but not required)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ChristusJoy/learning-zenml.git
cd learning-zenml
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install ZenML

```bash
pip install zenml
```

### 4. Initialize ZenML

```bash
zenml init
```

This creates a `.zen` directory to track your ZenML configuration.

### 5. (Optional) Start the ZenML Dashboard

```bash
zenml login --local
```

This launches a local dashboard where you can visualize your pipelines, runs, and artifacts.

---

## Learning Path

Follow these examples in order. Each builds on concepts from the previous one.

---

### Step 1: Hello Pipeline - Your First ZenML Pipeline

**File:** `hello_pipeline.py`

**Concepts Covered:**
- `@step` decorator - defines a unit of work
- `@pipeline` decorator - chains steps together
- Running a pipeline
- Accessing step outputs

**Code Walkthrough:**

```python
from zenml import pipeline, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def say_hello() -> str:
    logger.info("Executing say_hello step")
    return "Hello World!"

@pipeline
def hello_pipeline() -> str:
    message = say_hello()
    return message
```

**Key Takeaways:**
- A **step** is a Python function decorated with `@step` - it's the smallest unit of work
- A **pipeline** is a function decorated with `@pipeline` that orchestrates steps
- Steps must have **type hints** for inputs and outputs
- ZenML automatically tracks all executions

**Run It:**

```bash
python hello_pipeline.py
```

**What Happens:**
1. ZenML registers the pipeline and step
2. The step executes and returns "Hello World!"
3. The output is stored as an **artifact** in ZenML
4. You can retrieve the output programmatically

---

### Step 2: I/O Pipeline - Working with Inputs and Outputs

**File:** `io_pipeline.py`

**Concepts Covered:**
- Steps with multiple outputs
- Passing data between steps
- Using `Annotated` for named outputs
- Tuple unpacking for multiple returns

**Code Walkthrough:**

```python
from typing import Tuple
from typing_extensions import Annotated
from zenml import pipeline, step

@step
def load_data() -> Tuple[
    Annotated[list[int], "features"], 
    Annotated[list[int], "labels"]
]:
    return [1, 2, 3, 4], [1, 0, 1, 0]

@step
def count_rows(features: list[int], labels: list[int]) -> Annotated[int, "row_count"]:
    return len(features)

@pipeline
def io_pipeline() -> int:
    features, labels = load_data()
    row_count = count_rows(features, labels)
    return row_count
```

**Key Takeaways:**
- Use `Annotated[type, "name"]` to give outputs meaningful names
- Multiple outputs use `Tuple` with named annotations
- Outputs from one step become inputs to another
- ZenML tracks the **data lineage** - which step produced which data

**Run It:**

```bash
python io_pipeline.py
```

**What Happens:**
1. `load_data()` produces two artifacts: "features" and "labels"
2. Both artifacts are passed to `count_rows()`
3. The final "row_count" artifact is stored
4. All data dependencies are tracked automatically

---

### Step 3: Parameterized Pipeline - Configurable Pipelines

**File:** `param_pipeline.py`

**Concepts Covered:**
- Pipeline parameters
- Step parameters with defaults
- Configuring pipelines at runtime

**Code Walkthrough:**

```python
@step
def multiply(number: int, factor: int = 2) -> Annotated[int, "product"]:
    result = number * factor
    return result

@pipeline
def param_pipeline(number: int = 3, factor: int = 2) -> int:
    result = multiply(number=number, factor=factor)
    return result

# Run with custom parameters
run = param_pipeline(number=5, factor=10)
```

**Key Takeaways:**
- Pipelines can accept **runtime parameters**
- Steps can have **default values** for parameters
- Parameters are tracked with each run for reproducibility
- Different runs can use different configurations

**Run It:**

```bash
python param_pipeline.py
```

**Experiment:**
- Modify the `number` and `factor` values in the script
- Run multiple times and compare results in the dashboard

---

### Step 4: Cached Pipeline - Optimizing with Caching

**File:** `cache_pipeline.py`

**Concepts Covered:**
- Step caching with `enable_cache=True`
- How ZenML skips redundant computations
- When caching is triggered

**Code Walkthrough:**

```python
import time

@step(enable_cache=True)
def slow_step() -> Annotated[int, "answer"]:
    logger.info("🔄 Actually computing result... (sleeping 3 seconds)")
    time.sleep(3)
    return 42

@pipeline
def cache_pipeline():
    slow_step()

# First run - takes 3 seconds
cache_pipeline()

# Second run - instant (cached!)
cache_pipeline()
```

**Key Takeaways:**
- Caching prevents re-execution when inputs haven't changed
- First run executes fully; subsequent runs use cached results
- Caching is determined by: step code, input data, and parameters
- Saves time and compute resources in iterative development

**Run It:**

```bash
python cache_pipeline.py
```

**Observe:**
- Run 1: You'll see the "Actually computing..." message and 3-second delay
- Run 2: Instant completion - the step was skipped entirely!

---

### Step 5: Metadata Pipeline - Logging Metrics and Metadata

**File:** `meta_pipeline.py`

**Concepts Covered:**
- Logging metadata with `log_metadata()`
- Tracking metrics, hyperparameters, or any key-value data
- Viewing metadata in the dashboard

**Code Walkthrough:**

```python
from zenml import log_metadata, pipeline, step

@step
def compute_accuracy() -> Annotated[float, "accuracy_metric"]:
    acc = 0.93
    
    # Log metadata - visible in ZenML dashboard
    log_metadata({"accuracy": acc})
    
    return acc

@pipeline
def meta_pipeline() -> float:
    accuracy = compute_accuracy()
    return accuracy
```

**Key Takeaways:**
- `log_metadata()` attaches key-value pairs to steps/runs
- Metadata appears in the ZenML dashboard as cards
- Useful for logging: metrics, hyperparameters, dataset stats, model info
- Fully searchable and comparable across runs

**Run It:**

```bash
python meta_pipeline.py
```

**What to Check:**
- Open the ZenML dashboard
- Navigate to the run
- See the "accuracy" metadata card on the step

---

### Step 6: Tagged Pipeline - Organizing Artifacts with Tags

**File:** `tagged_pipeline.py`

**Concepts Covered:**
- Artifact configuration with `ArtifactConfig`
- Static tags on artifacts
- Dynamic tagging with `add_tags()`
- Cascade tags at pipeline level
- Using pandas DataFrames as artifacts

**Code Walkthrough:**

```python
import pandas as pd
from zenml import ArtifactConfig, Tag, add_tags, pipeline, step

@step
def create_raw_data() -> Annotated[
    pd.DataFrame, 
    ArtifactConfig(name="raw_data", tags=["raw", "input"])
]:
    data = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5],
        "feature_2": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0]
    })
    return data

@step
def process_data(raw_data: pd.DataFrame) -> Annotated[
    pd.DataFrame, 
    ArtifactConfig(name="processed_data", tags=["processed"])
]:
    processed = raw_data.copy()
    # Normalize features
    processed["feature_1"] = processed["feature_1"] / processed["feature_1"].max()
    processed["feature_2"] = processed["feature_2"] / processed["feature_2"].max()
    
    # Add dynamic tags
    add_tags(tags=["normalized", "ready_for_training"], infer_artifact=True)
    return processed

# Pipeline-level tags cascade to all artifacts
@pipeline(tags=["tutorial", Tag(name="experiment", cascade=True)])
def tagged_pipeline():
    raw_data = create_raw_data()
    processed_data = process_data(raw_data)
    return processed_data
```

**Key Takeaways:**
- `ArtifactConfig` lets you name artifacts and add static tags
- `add_tags()` adds tags dynamically during step execution
- `Tag(name="...", cascade=True)` applies tags to all artifacts in a pipeline
- Tags help organize, filter, and search artifacts
- ZenML handles pandas DataFrames (and many other types) automatically

**Run It:**

```bash
python tagged_pipeline.py
```

**Explore:**
- Check the dashboard for artifact tags
- Filter artifacts by tag in the UI
- Run multiple times to see tags accumulate

---

## ZenML Dashboard

The ZenML dashboard provides a visual interface to explore your ML workflows.

### Start the Dashboard

```bash
zenml login --local
```

### What You Can See

| Feature | Description |
|---------|-------------|
| **Pipelines** | All registered pipelines |
| **Runs** | Execution history with status |
| **DAG View** | Visual step dependencies |
| **Artifacts** | All produced data with lineage |
| **Metadata** | Logged metrics and info |
| **Tags** | Filter and organize artifacts |

---

## Key Concepts Summary

| Concept | Description | Example |
|---------|-------------|---------|
| **Step** | Single unit of work | `@step def my_step(): ...` |
| **Pipeline** | Chain of steps | `@pipeline def my_pipeline(): ...` |
| **Artifact** | Data produced by steps | Return values from steps |
| **Annotated** | Named outputs | `Annotated[int, "count"]` |
| **Caching** | Skip redundant work | `@step(enable_cache=True)` |
| **Metadata** | Key-value tracking | `log_metadata({"key": value})` |
| **Tags** | Organize artifacts | `ArtifactConfig(tags=["tag1"])` |

---

## Recommended Learning Order

```
1. hello_pipeline.py    → Understand @step and @pipeline basics
         ↓
2. io_pipeline.py       → Learn data flow between steps
         ↓
3. param_pipeline.py    → Make pipelines configurable
         ↓
4. cache_pipeline.py    → Optimize with caching
         ↓
5. meta_pipeline.py     → Track metrics and metadata
         ↓
6. tagged_pipeline.py   → Organize with tags (advanced)
```

---

## Next Steps

After completing this tutorial, Checkout my [Vehicle insuracne ZenMl](https://github.com/ChristusJoy/vehicle-insurance-zenml) project which explores an end to end pipeline for a sample project.

1. **Stacks & Components** - Deploy to cloud infrastructure
2. **Model Registry** - Version and manage models
3. **Experiment Tracking** - Integrate with MLflow, W&B
4. **Deployment** - Serve models in production
5. **Custom Materializers** - Handle custom data types

### Resources

- [ZenML Documentation](https://docs.zenml.io/)
- [ZenML GitHub](https://github.com/zenml-io/zenml)
- [ZenML Slack Community](https://zenml.io/slack)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

