# pgvector 公式イメージに、日本語の全文検索用に pg_bigm（2-gram）をビルドして追加する
FROM pgvector/pgvector:pg17

ARG PG_BIGM_VERSION=v1.2-20250903

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates postgresql-server-dev-17 \
    && mkdir /tmp/pg_bigm \
    && curl -fsSL "https://github.com/pgbigm/pg_bigm/archive/refs/tags/${PG_BIGM_VERSION}.tar.gz" \
        | tar -xz -C /tmp/pg_bigm --strip-components=1 \
    && make -C /tmp/pg_bigm USE_PGXS=1 \
    && make -C /tmp/pg_bigm USE_PGXS=1 install \
    && apt-get purge -y --auto-remove build-essential curl postgresql-server-dev-17 \
    && rm -rf /var/lib/apt/lists/* /tmp/pg_bigm
