#!/bin/bash
#SBATCH --account={{ACCOUNT}}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{NUM_CPUS}}
#SBATCH --gres=gpu:{{GPU_TYPE}}{{NUM_GPUS}}
#SBATCH --mem={{MEM}}
#SBATCH --time={{TIME}}
#SBATCH --output={{LOG_FILE}}
#SBATCH --job-name={{RUN_ID}}-{{EXPERIMENT_ID}}

mkdir -p {{OUTPUT_DIR_IN_CLUSTER}}
mkdir -p {{LOG_DIR}}

module load {{MODULES_TO_LOAD}}

# 1. Copy your container on the compute node
rsync -avz {{IMAGE_PATH}} $SLURM_TMPDIR

# 2. Executing your code with Apptainer
export PYTHONPATH=/src:$PYTHONPATH
# Clean stale __pycache__ to force recompilation from current .py files
find {{CODE_DIR_IN_CLUSTER}} -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
{{SINGULARITY_COMMAND}} exec --nv --writable-tmpfs \
        --env PYTHONPATH=/src \
        --env PYTHONUNBUFFERED=1 \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --bind {{CODE_DIR_IN_CLUSTER}}:/src \
        --bind {{OUTPUT_DIR_IN_CLUSTER}}:{{OUTPUT_DIR_IN_CONTAINER}} \
        $SLURM_TMPDIR/{{IMAGE_NAME}} \
        bash -c "cd /src && {{COMMAND}}"
