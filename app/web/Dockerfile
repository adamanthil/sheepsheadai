# syntax=docker/dockerfile:1
# Build from web/: docker build web
# NEXT_PUBLIC_API_BASE is inlined at BUILD time. The default (empty) means
# same-origin — correct behind the reverse proxy — so one image works on any
# domain without rebuilding.

FROM node:26.13.1-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:26.13.1-alpine AS build
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
# Source maps go to Sentry when the build is given the auth token as a
# BuildKit secret (CI: --secret id=sentry_auth_token,env=SENTRY_AUTH_TOKEN),
# and are deleted after upload (next.config.mjs). A secret, not an ARG, so
# the token stays out of the image and its history; without it the build
# skips the upload.
RUN --mount=type=secret,id=sentry_auth_token,env=SENTRY_AUTH_TOKEN \
    npm run build

FROM node:26.13.1-alpine AS run
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
