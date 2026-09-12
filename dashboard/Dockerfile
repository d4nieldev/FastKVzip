# Stage 1: build the dashboard bundle.
FROM node:22-alpine AS web
WORKDIR /build
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
# vite.config.ts writes to ../server/static, which is outside this stage's
# WORKDIR, so build to a local dist and copy it across explicitly.
RUN npx vite build --outDir dist --emptyOutDir

# Stage 2: the server, with the bundle baked in.
FROM python:3.12-slim
WORKDIR /app

COPY server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server/app ./app
COPY --from=web /build/dist ./static

ENV DATA_DIR=/data \
    STATIC_DIR=/app/static \
    PORT=8000
# Only log files live here, and losing them costs nothing: the agent re-ships
# every log from byte 0 after a wipe (see logstore.append). Records nothing can
# replace -- projects, what is filed in them, chosen colours, read marks -- are
# in Postgres, which is why DATABASE_URL is required.
VOLUME ["/data"]
EXPOSE 8000

# PaaS platforms inject their own $PORT.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
