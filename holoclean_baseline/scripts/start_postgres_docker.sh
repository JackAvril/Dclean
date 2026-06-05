#!/usr/bin/env bash
set -e
docker rm -f pghc 2>/dev/null || true
docker run --name pghc \
  -e POSTGRES_DB=holo \
  -e POSTGRES_USER=holocleanuser \
  -e POSTGRES_PASSWORD=abcd1234 \
  -p 5432:5432 \
  -d postgres:11
echo "[OK] PostgreSQL started: db_name=holo db_user=holocleanuser db_pwd=abcd1234 db_host=localhost"
