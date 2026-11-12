# Research compute environment (twin solver, PhysicsNeMo, Warp, tests) for a CUDA host.
# Reproduces the conda env `hydro` of the original GPU server from its pinned package list.
#
# Build context is the repository root:
#   docker compose -f deploy/host/docker-compose.yml --profile gpu build compute
#
# The image carries only the Python stack and an editable install of the package. At run time
# the whole checkout (code, data/, results/, chou-shui-data/, ...) is bind-mounted over /work,
# so the container always runs the host's live code and data; nothing private is baked in.
# GPU access needs the NVIDIA driver on the host plus the NVIDIA Container Toolkit; the CUDA
# runtime itself comes from the torch cu128 wheels (driver >= 525).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git tmux procps ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY deploy/host/requirements-gpu.lock.txt /tmp/requirements-gpu.lock.txt
RUN pip install -r /tmp/requirements-gpu.lock.txt

# Editable install: setuptools maps the packages to /work, which the bind mount replaces with
# the live checkout. AMP_V2/ and dataprep_staging/ are plain directories on PYTHONPATH.
WORKDIR /work
COPY pyproject.toml README.md /work/
COPY hydrophysics /work/hydrophysics
COPY hydrophysics_analyst /work/hydrophysics_analyst
RUN pip install --no-deps -e . && rm -rf /work/*

# Runs as the host user (compose passes UID:GID) so files written to the mounted checkout keep
# the host owner; HOME is a world-writable directory for the Warp / matplotlib caches.
RUN mkdir -p /home/runner && chmod 1777 /home/runner
ENV HOME=/home/runner PYTHONPATH=/work MPLBACKEND=Agg

CMD ["bash"]
