# Hugging Face Space (sdk: docker) for the twin analyst: chat beside the decision page.
# Builds one pinned commit of the public repo; deploy.py fills REPO_SHA before upload.
# No torch, no GPU: the analyst needs numpy, pandas and anthropic.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user

ARG REPO_SHA=main
RUN git init -q app && cd app \
    && git remote add origin https://github.com/Rekin226/HydroPhysicsAI.git \
    && git fetch -q --depth 1 origin "$REPO_SHA" && git checkout -q FETCH_HEAD \
    && git log -1 --format='built from %H'

WORKDIR /home/user/app
RUN pip install --no-cache-dir --user -e ".[analyst]" && mkdir -p results/analyst/audit
COPY --chown=user launch.py /home/user/launch.py

EXPOSE 7860
CMD ["python", "/home/user/launch.py"]
