# ODEForge container.
#
# Versions are pinned so the image reproduces the numbers in README.md. The lock file
# is installed directly; nothing is fetched at run time.

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so the layer caches independently of the source.
COPY requirements.lock.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.lock.txt

COPY . .

# Fail fast and loudly if the import graph is broken.
RUN python -c "import core, data, integrators, routing, autodiff, training, hpo, eval, pipeline, cli"

# Default: the end-to-end demonstration.
CMD ["python", "examples/run_demo.py"]
