# PREDATOR Agent - Dockerfile
# Multi-stage build for minimal production image

# ── Build stage ─────────────────────────────────────────────────────────────
FROM node:20-slim AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --omit=dev

COPY . .

# ── Production stage ────────────────────────────────────────────────────────
FROM node:20-slim

LABEL maintainer="Based on Agentic Theory by Justo Tapiador García (UA)"
LABEL description="PREDATOR – Deep Agentic AI System (AJN Framework)"
LABEL version="2.0.0"

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends dumb-init && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r predator && useradd -r -g predator -d /app -s /sbin/nologin predator

WORKDIR /app

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/src ./src
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/web ./web
COPY --from=builder /app/config ./config
COPY --from=builder /app/package.json ./

# Create data directories
RUN mkdir -p /app/data/memory /app/checkpoints /app/logs && \
    chown -R predator:predator /app

USER predator

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD node -e "fetch('http://localhost:3000/api/status').then(r => r.ok ? process.exit(0) : process.exit(1)).catch(() => process.exit(1))"

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "web/server.js"]
