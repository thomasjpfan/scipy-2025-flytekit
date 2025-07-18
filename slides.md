title: Dive into Flytekit's Internals: A Python SDK to Quickly Bring your Code Into Production
use_katex: False
class: title-slide

# Dive into Flytekit's Internals
## A Python SDK to Quickly Bring your Code Into Production

![:scale 45%](images/flyte.png)

.g.g-middle.g-center[
.g-8[
.larger[Thomas J. Fan]<br>
<a href="https://www.github.com/thomasjpfan" target="_blank" class="title-link"><span class="icon icon-github right-margin"></span>@thomasjpfan</a>
<a href="http://linkedin.com/in/thomasjpfan" target="_blank" class="this-talk-link">linkedin.com/in/thomasjpfan</a>
<a class="this-talk-link", href="https://github.com/thomasjpfan/scipy-2025-flytekit" target="_blank">github.com/thomasjpfan/scipy-2025-flytekit</a>
]
.g-4[
![:scale 90%](images/qrcode.png)
]
]

---

# Introduction

.g.g-middle[
.g-6[
## OSS: Scikit-learn maintainer
]
.g-6.g-center[
![:scale 50%](images/scikit-learn-logo-without-subtitle.svg)
]
]

.g.g-middle[
.g-6[
## Member of Technical Staff @ Modal
]
.g-6.g-center[
![:scale 80%](images/modal.png)
]
]


.g.g-middle[
.g-6[
## Worked on Flyte & Python SDK
]
.g-6.g-center[
![:scale 70%](images/flyte.png)
]
]

---

# Contents

.g.g-middle[
.g-8.larger[
## Why Flyte? 🛩️
## From Python to Remote 🛜
## Unraveling a Flyte Task 🧶
]
.g-4.g-center[
![:scale 100%](images/toc.jpg)
]
]

---


# Why Flyte? 🛩️

.g.g-middle[
.g-5.larger[
## Reliable 🪢
## Scalable 🗻
## Iterate Fast 🏎️
]
.g-7.g-center[
![](images/flyte-linux.png)
]
]

---

# High level overview of Flyte ✈️

<br>

![](images/python-dag-cluster.png)

---

class: top

# Python Code to Static Workflow 🐍

## Task

```python
from flytekit import task, workflow

@task
def load_data() -> pd.DataFrame:
	...

@task
def preprocess(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	...
```

--

class: top

## Workflow 👟

```python
@workflow
def main() -> float:
    data = load_data()
    train, test = preprocess(data=data)
    model = train_model(train=train)
    return evaluate_model(model=model, data=test)
```

---

class: top

<br>

# Python Code to Static Workflow 🐍

```python
@workflow
def main() -> float:
    data = load_data()
    train, test = preprocess(data=data)
    model = train_model(train=train)
    return evaluate_model(model=model, data=test)
```

--

## CLI

```bash
pyflyte run --remote main.py main
```

--

![](images/workflow.jpg)

---

# Workflow: Serialized with Protobuf 🏞️
## Workflow can now be managed by Golang

![](images/workflow.jpg)

---

class: top

<br>

# Workflow to Kubernetes 🐳

```python
@task
def preprocess(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	...
```

<br>

--

## Pod Specification

```yaml
Name: azthpf845mgkp7f6hkpw-n1-0
Namespace: flytesnacks-development
Containers:
  azthpf845mgkp7f6hkpw-n1-0:
    Args:
	  pyflyte-fast-execute
	  --additional-distribution s3://my-s3-bucket/...
```

---

# Flyte @ High Level 🌎
## From Local to Remote

<br>

![!scale 100%](images/python-dag-cluster.png)

---

class: chapter-slide

# Deep Dive into Tasks 🤿

---

class: top

<br>

# Resources (CPU & Memory) 💻

```python
from flytekit import Resources

@task(
*    requests=Resources(cpu=2, mem="2Gi")
*    limits=Resources(cpu=4, mem="8Gi"),
)
def train_model(train: pd.DataFrame) -> BaseEstimator:
	...
```

--

## Pod Spec

```yaml
Requests:
  cpu: 2
  memory: 2Gi
Limits:
  cpu: 4
  memory: 8Gi
```

---

# Resources: GPU 🏎️

```python
from flytekit.extras.accelerators import A100

@task(
*   accelerator=A100
)
def train_model(...):
	...
```

# Pod Spec

```yaml
tolerations:
  - operator: Equal
    value: nvidia-tesla-a100
	effect: NoSchedule
```

---

# Python Dependencies 🐍

## Prebuilt image

```python
@task(
	container_image="ghcr.io/flyteorg/flytekit:py3.12-1.16.1"
)
def train_model(...):
    ...
```

---


# Python Dependencies (ImageSpec) 🐍

.g.g-middle[
.g-6[
```python
from flytekit import ImageSpec

image = ImageSpec(
	packages=["numpy", "scikit-learn"],
    registry="ghcr.io/thomasjpfan"
)

@task(
*   container_image=image
)
def train_model(...):
    ...
```
]
.g-6.g-center[
![](images/docker_logo.png)
]
]

---

class: top

<br>

# Getting Local Code to Remote 🛜
### "Fast Registration"

```python
*from utils import split_data, create_features

@task
def preprocess(data: pd.DataFrame) ->  tuple[pd.DataFrame, pd.DataFrame]:
    featured_data = create_features
	train, test = split_data(featured_data)
    return train, tes
```

--

### Folder structure

```bash
utils.py
wf.py
```

--

### CLI

```bash
pyflyte run --remote wf.py main
```

- Uploads to Object store (S3)

---

class: top

<br>

# Binary files

### Folder structure

```bash
bin/custom_executable
wf.py
```

--

### Task code

```python
@task
def run_executable():
    run(["bin/custom_executable"], text=True)
```

--

### CLI

```bash
pyflyte run --remote --copy all wf.py main
```

- Uploads to Object store (S3)

---

class: top

# How does container know about the code?

### Entrypoint in Pod Spec:

```yaml
Args:
  pyflyte-fast-execute
  --additional-distribution
* s3://my-s3-bucket/flytesnacks/development/...
  --dest-dir .
```

--

### Which module to load?

```python
pyflyte run --remote wf.py preprocess
```

--

### Entrypoint

```yaml
  pyflyte-execute ...
  --resolver flytekit.core.python_auto_container.default_task_resolver
  --
* task-module wf
* task-name preprocess
```

---

# Strict Typing 🔥
## Types must match!

```python
@task
*def add_one(x: int) -> int:
    return x + 1


@workflow
*def wf(x: str):
    add_one(x=x)
```

---

class: top

<br><br>

# Literal Types

```python
@task
def say_hello(name: str) -> str:
    return f"Hello, {name}!"

@workflow
def hello_world_wf(name: str = "world") -> str:
    greeting = say_hello(name=name)
    return greeting
```

--

### Converted directly to "Flyte Literal" types

- "Simple Python types": `str`, `int`, `bool`, `float`
- Stored in object store (S3) as "metadata"

---

class: top

<br>

# Dataclass-like types

```python
from dataclasses import dataclass
from flytekit import task

@dataclass
class MyData:
    name: str
    value: int

@task
def create_data() -> MyData:
    return MyData(name="abc", value=123)
```

--

### Serialized with `MessagePack`
- Stored in object store (S3)

---

class: top

# File type: `FlyteFile` 🗒️
- Stored to Object store (S3)

--

```python
from flytekit import FlyteFile

@task
def create_file() -> FlyteFile:
    file = FlyteFile.new_remote_file()

    with file.open("w") as f:
        f.write("my content")
    return file
```

--

### Downloaded when used

```python
@task
def read_file(file: FlyteFile) -> str:
    with file.open("r") as f:
        return f.read()
```

---

class: top

<br>

# Directory type: `FlyteDirectory` 🗃️
- Stored to Object store (S3)

--

```python
from flytekit import FlyteDirectory

@task
def create_directory() -> FlyteDirectory:
    """Create directory of files and return it"""
```

--

### Explicit Download

```python
@task
def read_directory(directory: FlyteDirectory):
    local_dir = directory.download()
    # Use data in load_dir
```

---

class: top

# Types in `flytekit`

```python
@task
def make_dataset() -> np.ndarray:
	return np.asarray([[1, 2, 3], [4, 5, 6]])
```

--

### Type Transformer!

```python
class NumpyArrayTransformer(AsyncTypeTransformer[np.ndarray]):
*   async def async_to_literal(self, ...);
		# Serialize data
	    np.save(...)
        return Literal(...)

*   async def async_to_python_value(self, ...):
        # Deserialize data
        return np.load(...)
```

--

### Plugin Registration

```python
from flytekit.core.type_engine import TypeEngine

TypeEngine.register(NumpyArrayTransformer())
```

[flytekit/types/numpy/ndarray.py](https://github.com/flyteorg/flytekit/blob/master/flytekit/types/numpy/ndarray.py)

---

class: top

<br>

# Extending with Custom Type 🐻‍❄️

```python
import polars as pl

@task
def preprocess(df: pl.DataFrame) -> pl.DataFrame:
	...
```

--

<br>

### Installed with a Plugin

```bash
pip install flytekitplugins-polars
```
- Serialized into parquet files

---

class: top

<br>

# How does Plugins get load? 🐻‍❄️

### Entrypoint!

```python
setup(
    entry_points={"flytekit.plugins": ["polars=flytekitplugins.polars"]},
)
```

--

<br>

### Registration during loading

```python
StructuredDatasetTransformerEngine.register(PolarsDataFrameToParquetEncodingHandler())
StructuredDatasetTransformerEngine.register(ParquetToPolarsDataFrameDecodingHandler())
```

[plugins/flytekit-polars/flytekitplugins/polars/sd_transformers.py](https://github.com/flyteorg/flytekit/blob/master/plugins/flytekit-polars/flytekitplugins/polars/sd_transformers.py)

---

# Data is Stored to Object store

.center[
![:scale 60%](images/task-s3.png)
]

---

class: top

<br>

# How is data passed between tasks? 🔌

![](images/workflow.jpg)

### Task runs in Different Containers 🐳
### Data in Object Store 🏪
### Location Changes Between Executions 🔄

---

# Dynamic Entrypoints with a Template 🖨️

```python
class PythonAutoContainerTask(...):
    def get_default_command(self, settings):
        container_args = [
            "pyflyte-execute",
            "--inputs",
            "{{.input}}",
            "--output-prefix",
            "{{.outputPrefix}}",
            "--raw-output-data-prefix",
            "{{.rawOutputDataPrefix}}",
            ...
        ]
        return container_args
```

.footnote-back[
[flytekit/core/python_auto_container.py](https://github.com/flyteorg/flytekit/blob/master/flytekit/core/python_auto_container.py)
]


---

class: top

# Dynamic Entrypoints with a Template 🖨️

```yaml
Args:
  pyflyte-fast-execute
  -additional-distribution s3://...
  --
  pyflyte-execute
* --inputs {{.inputs}}
* --output-prefix {{.outputPrefix}}
```

--

## Entrypoint is Populated by Flyte

```yaml
Args:
  pyflyte-fast-execute
  -additional-distribution s3://...
  --
  pyflyte-execute
* --inputs s3://my-s3-bucket/metadata/.../data/inputs.pb
* --output-prefix s3://my-s3-bucket/metadata/.../data/0
```

---

class: top

<br>

# Entrypoint is Populated by Flyte ✈️

```yaml
Args:
  pyflyte-fast-execute
  -additional-distribution s3://...
  --
  pyflyte-execute
* --inputs s3://my-s3-bucket/metadata/.../data/inputs.pb
* --output-prefix s3://my-s3-bucket/metadata/.../data/0
```

### Serialized into `inputs.pb` & `outputs.pb`

- Inputs: `s3://my-s3-bucket/metadata/.../data/inputs.pb`
- **Output Prefix**: `s3://my-s3-bucket/metadata/.../data/0`
- Outputs: `s3://my-s3-bucket/metadata/.../data/0/outputs.pb`


---

class: top

# What about errors? ⚠️

.center[
![:scale 90%](images/error.png)
]

--

- Errors: `s3://my-s3-bucket/metadata/.../data/0/errors.pb`

---

class: top

<br>

# Flyte Deck

.g.g-middle[
.g-8[
```python
from flytekitplugins.deck.renderer import (
    FrameProfilingRenderer
)

@task(enable_deck=True)
def create_deck():
    # Create HTML snippet
    df = pd.DataFrame(...)
    Deck(
        "Frame Rendered",
*       FrameProfilingRenderer().to_html(df=df)
    )
```
]
.g-4.g-center[
![:scale 100%](images/deck.jpg)
]
]

--

- **Output Prefix**: `s3://my-s3-bucket/metadata/.../data/0`
- Static HTML: `s3://my-s3-bucket/metadata/.../data/0/deck.html`

---

# Flyte Deck (With Types)

.g.g-middle[
.g-8[
```python
class DataFrameSummaryRenderer:
*   def to_html(self, df: pd.DataFrame) -> str:
        # Creates HTML from df

@task(enable_deck=True)
def create_deck_with_typing() -> (
*   Annotated[
*       pd.DataFrame, DataFrameSummaryRenderer()
*   ]
):
    df = pd.DataFrame(...)
    return df
```
]
.g-4.g-center[
![:scale 100%](images/deck.jpg)
]
]

---

# Retries and caching 📖

![](images/workflow-detailed.jpg)

.g[
.g-6[
### Retries

```python
@task(
*   retries=5,
)
def preprocess(input: pd.DataFrame):
	...
```
]
.g-6[
]
]

---

# Retries and caching 📖

![](images/workflow-detailed.jpg)

.g[
.g-6[
### Retries

```python
@task(
*   retries=5,
)
def preprocess(input: pd.DataFrame):
	...
```
]
.g-6[
### Caching

```python
@task(
*   cache=True, cache_version="v1"
)
def preprocess(input: pd.DataFrame):
    ...
```
]
]

---

class: top

<br>

# Language agnostic design! 🗺️

- Metadata stored as `inputs.pb`, `outputs.pb` & `errors.pb` in S3/GCS/Minio
- Raw Data (Model weights, etc) are stored in S3/GCS/Minio

--

## Shell Script

```python
t2 = ShellTask(
    name="task_2",
    script="""
    set -ex
    cp {inputs.x} {inputs.y}
    tar -zcvf {outputs.j} {inputs.y}
    """,
*   inputs=kwtypes(x=FlyteFile, y=FlyteDirectory),
*   output_locs=[OutputLocation(var="j", var_type=FlyteFile, location="{inputs.y}.tar.gz")],
)
```

---

class: top

<br>

# Dynamic Runtime variables 👟

```python
from flytekit import current_context

@task
def query_environment():
	ctx = current_context()

    print(ctx.execution_id.name)
    print(ctx.execution_id.domain)
    print(ctx.execution_id.project)
```

- Useful for sending execution data to another service

--

## Kubernetes Pod Specification

```yaml
Environment:
  FLYTE_INTERNAL_EXECUTION_ID: a45gqlrs7c87dqkksbrl
  FLYTE_INTERNAL_EXECUTION_PROJECT: flytesnacks
  FLYTE_INTERNAL_EXECUTION_DOMAIN: development
```

---

# PodTemplates: Full Kubernetes Control 🐳

```python
from flytekit import PodTemplate
from kubernetes.client import V1PodSpec, V1Container, V1Volume, V1Toleration

pod_template = PodTemplate(
	primary_container_name="primary",
	labels={"key1": "value1", "key2": "value2"},
	annotations={"key3": "value3"},
	pod_spec=V1PodSpec(...),
	volumes=[V1Volume(name="volume")],
	tolerations=[
		V1Toleration(...)
	]
)

@task(
*   container_image=pod_template
)
def my_task():
	...
```
---

# Scaling up! 🆙


```python
from flytekit import map_task

@workflow
def scale_map_task():
    datasets = query_many_datasets()
    results = map_task(preprocess)(data=datasets)
```

.center[
![:scale 60%](images/map_task.png)
]

---

class: top

# Ray, Dask, Spark
## Running on your Cluster!

.g.g-middle.g-center[
.g-4[
![](images/dask.jpg)
]
.g-4[
![](images/ray.png)
]
.g-4[
![:scale 90%](images/spark.png)
]
]

--


.g.g-middle[
.g-6[
### Powered by Kubernetes Operators
- Ray: [kuberay](https://github.com/ray-project/kuberay)
- Dask: [dask-kubernetes](https://github.com/dask/dask-kubernetes)
- Spark: [spark-operator](https://github.com/kubeflow/spark-operator)
]
.g-6[
![](images/kubernetes-logo.jpg)
]
]


---

class: top

# Dask

.g.g-middle[
.g-8[
```python
from flytekit import Resources, task
from flytekitplugins.dask import (
    Dask, Scheduler, WorkerGroup
)

@task(
* task_config=Dask(
      scheduler=Scheduler(
          limits=Resources(cpu="1", mem="2Gi"),
      ),
      workers=WorkerGroup(
          limits=Resources(cpu="4", mem="10Gi"),
      ),
  ),
)
def dask_preprocessing():
	...
```
]
.g-4[
![](images/dask.jpg)
]
]

### Install Plugin

```bash
pip install flytekitplugins-dask
```

---

# Under the Hook (Dask)

![](images/dask-workflow.png)

---

class: top

<br>

# How does `task_config=Dask(...)` work? 🤔

```python
@task(task_config=Dask(...))
def dask_preprocessing():
```

--

```python
@dataclass
class Dask:
    scheduler: Scheduler
    workers: WorkerGroup
```

--

## Declare Resources in `DaskTask`

```python
class DaskTask(PythonFunctionTask[Dask]):
    def get_custom(self, settings) -> Dict[str, Any]:
        # construct dictionary representing resources specified in the Dask dataclass

TaskPlugins.register_pythontask_plugin(Dask, DaskTask)
```

.footnote-back[
[plugins/flytekit-dask/flytekitplugins/dask/task.py](https://github.com/flyteorg/flytekit/blob/master/plugins/flytekit-dask/flytekitplugins/dask/task.py)
]

---

# Spark ✨

.g.g-middle[
.g-8[
```python
from flytekitplugins.spark import Spark

spark_config = Spark(
    spark_conf={
        "spark.driver.memory": "1000M",
        "spark.executor.memory": "1000M",
        "spark.executor.cores": "1",
        "spark.executor.instances": "2",
        "spark.driver.cores": "1",
        "spark.jars": "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar",
    }
)

@task(
*   task_config=spark_config
)
def hello_spark(partitions: int) -> float:
    ...
```
]
.g-4[
![](images/spark.png)
]
]

---

# Ray

.g.g-middle[
.g-8[
```python
from flytekitplugins.ray import (
    HeadNodeConfig, RayJobConfig, WorkerNodeConfig
)

ray_config = RayJobConfig(
    head_node_config=...,
    worker_node_config=...,
    runtime_env={"pip": ["numpy", "pandas"]},
)

@task(
*   task_config=ray_config,
    requests=Resources(mem="2Gi", cpu="2"),
    container_image=custom_image,
)
def ray_task(n: int) -> typing.List[int]:
    ...
```
]
.g-4[
![](images/ray.png)
]
]

---

# Pytorch Elastic Training  🔥

.g.g-middle[
.g-8[
```python
from flytekitplugins.kfpytorch import Elastic

@task(
* task_config=Elastic(
    nnodes=2,
    nproc_per_node=4,
  ),
)
def train():
    ...
```
]
.g-4[
![](images/pytorch.png)
]
]

---

# High level overview of Flyte ✈️

<br>

![](images/python-dag-cluster.png)

---

# Backend architecture 🔬

.center[
![:scale 65%](images/flytepropeller-architecture.png)
]

.footnote-back[
[union.ai/docs/flyte/architecture/component-architecture/flytepropeller_architecture/](https://www.union.ai/docs/flyte/architecture/component-architecture/flytepropeller_architecture/)
]

---


.g.g-middle[
.g-8[
# Why Flyte?

## Reliable 🪢
- Build on top of **Kubernetes**
- Static Workflow Graphs & Embracing Python Typing

## Scalable 🗻
- GPUs + Dask, Ray, Spark, PyTorch Distributed
- `map_task` for Parallelism

## Iterate Fast 🏎️
- Local Python code to Remote
- Recover from failures
]
.g-4.g-center[
![](images/flyte.png)
]
]

---

class: title-slide

# Dive into Flytekit's Internals
## A Python SDK to Quickly Bring your Code Into Production

![:scale 45%](images/flyte.png)

.g.g-middle.g-center[
.g-8[
.larger[Thomas J. Fan]<br>
<a href="https://www.github.com/thomasjpfan" target="_blank" class="title-link"><span class="icon icon-github right-margin"></span>@thomasjpfan</a>
<a href="http://linkedin.com/in/thomasjpfan" target="_blank" class="this-talk-link">linkedin.com/in/thomasjpfan</a>
<a class="this-talk-link", href="https://github.com/thomasjpfan/scipy-2025-flytekit" target="_blank">github.com/thomasjpfan/scipy-2025-flytekit</a>
]
.g-4[
![:scale 80%](images/qrcode.png)
]
]
