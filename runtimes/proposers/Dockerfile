FROM node:22.23.0-bookworm-slim
ARG CODEX_VERSION=0.165.0
ARG CLAUDE_CODE_VERSION=2.1.312
RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @openai/codex@${CODEX_VERSION} @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}
USER node
WORKDIR /workspace
