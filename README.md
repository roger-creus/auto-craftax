# skills-via-llm

# Installation

Build the Docker container:

```bash
docker build -t skill:latest .
```

## Test your installation:

You can either play with an interactive shell:

```bash
docker run --rm --gpus all -it -v $(pwd):/workspace -w /workspace skill:latest /bin/bash
```

or try running the test commands (some can take a while because craftax loads all textures everytime inside docker!)

```bash
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace skill:latest python tests/test_jax_and_torch.py
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace skill:latest python tests/test_craftax_vanilla.py
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace skill:latest python tests/test_craftax_torch_wrapper.py
```

