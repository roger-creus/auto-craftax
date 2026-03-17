CONDA_OVERRIDE_CUDA=12.2 conda create --strict-channel-priority -c conda-forge -n skill-rl jaxlib=*=*cuda* jax python=3.10 -y
conda activate skill-rl
pip install craftax
pip install brax
pip install -r requirements.txt
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .