#!/usr/bin/env bash
# Creates the shadow database used by graphile-migrate so the dev container
# is ready to run migrations on first boot.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-SQL
    CREATE DATABASE sheepshead_shadow OWNER $POSTGRES_USER;
SQL
