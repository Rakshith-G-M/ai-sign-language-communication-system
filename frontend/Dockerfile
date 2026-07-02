# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – Build
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (layer-cached if package files don't change)
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

# Copy source and build the production bundle
COPY . .
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Serve
# ─────────────────────────────────────────────────────────────────────────────
FROM nginx:alpine AS runner

# Remove the default nginx placeholder
RUN rm -rf /usr/share/nginx/html/*

# Copy the Vite build output
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx config: serve the SPA and forward /api/* to the backend
COPY nginx.spa.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
