# AgentShield frontend image — runs the same `next dev` dev server used locally
# (npm run dev), so behavior matches run_all.sh exactly.
#
# The only Docker-specific adjustment is binding to 0.0.0.0 instead of Next's default
# host, via the CMD below (no package.json / source change) — required so the container's
# published port is reachable from the host browser.
FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 3100

CMD ["npx", "next", "dev", "-H", "0.0.0.0", "-p", "3100"]
