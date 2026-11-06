# Build from web/: docker build web
# NEXT_PUBLIC_API_BASE is inlined at BUILD time. The default (empty) means
# same-origin — correct behind the reverse proxy — so one image works on any
# domain without rebuilding.

FROM node:22.23.4-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22.23.4-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_BASE=
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE
# Names the build, so a page left open from an older deploy reloads on its
# next navigation instead of soft-navigating on stale JavaScript. CI passes
# the image's content tag (app/scripts/deploy_tags.sh); empty, as in a local
# build, leaves it unset.
ARG NEXT_DEPLOYMENT_ID=
ENV NEXT_DEPLOYMENT_ID=$NEXT_DEPLOYMENT_ID
RUN npm run build

FROM node:22.23.4-alpine AS run
WORKDIR /app
ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    PORT=3000
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
# Standalone output does not bundle public/; without this the app icons that
# metadata and the web manifest point at 404 in the deployed image.
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
