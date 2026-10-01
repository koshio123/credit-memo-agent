-- 文書・チャンク・埋め込み。CREATE ... IF NOT EXISTS なので、何度実行してもよい。
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id     text PRIMARY KEY,
    sec_code   text NOT NULL,
    company    text NOT NULL,
    period_end text NOT NULL,
    n_pages    integer NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     text PRIMARY KEY,
    doc_id       text NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    seq          integer NOT NULL,
    heading_path text[] NOT NULL,
    page_start   integer NOT NULL,
    page_end     integer NOT NULL,
    text         text NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id, seq);

-- 埋め込みはモデルごとに次元が違うので、次元を固定しない vector にする。
-- 検索は必ず model で絞り込む。件数が少ない（数千）ので、厳密な全件検索で足りる。
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id  text NOT NULL REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    model     text NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
