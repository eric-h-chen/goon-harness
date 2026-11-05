FROM node:22.23.0-bookworm-slim
WORKDIR /runtime
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts --omit=dev && npm cache clean --force
COPY worker.mjs ./
USER node
ENTRYPOINT ["node", "/runtime/worker.mjs"]
