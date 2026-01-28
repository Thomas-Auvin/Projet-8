# Dockerfile (HF Spaces - Docker SDK / prod-like)
FROM python:3.13-slim-bookworm

# 1) Dépendance système requise par LightGBM (OpenMP runtime)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2) Installer uv (binaire) via l'image officielle Astral (rapide, reproductible)
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

# 3) User id 1000 (recommandé sur Hugging Face Spaces)
RUN useradd -m -u 1000 user

# 4) Dossiers de travail + persistance (/data) avec droits OK
RUN mkdir -p /home/user/app /data \
    && chown -R user:user /home/user /data \
    && chmod 777 /data

USER user
ENV HOME=/home/user \
    UV_LINK_MODE=copy \
    P8_DB_PATH=/data/predictions.sqlite \
    P8_ARTIFACTS_DIR=/home/user/app/app/artifacts \
    P8_STRICT_INPUT=1

WORKDIR $HOME/app

# 5) Dépendances d'abord (meilleur cache Docker)
COPY --chown=user:user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 6) Code ensuite
COPY --chown=user:user . .

# HF Spaces: 7860 par convention
EXPOSE 7860
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
