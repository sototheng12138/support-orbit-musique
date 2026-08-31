# Source provenance and access boundary

Frozen on 2026-08-14 before any model training.

## Official source

- Project: MuSiQue: Multi-hop Questions via Single-hop Question Composition (TACL 2022)
- Repository: `https://github.com/StonyBrookNLP/musique.git`
- Commit: `922ac98f19a201998dbdae6d7f2887a5258dbdeb`
- License: CC BY 4.0
- Local `LICENSE` SHA256: `cce5d01fa4a83b794271bd2c28cffdf99afd43c803e6ddefddae39b591ea7448`
- Official Google Drive file ID: `1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h`

## Downloaded archive

- Path: `/home/hesong/AI-Agent-Projects/data/musique_official/musique_v1.0.zip`
- Size: `272049578` bytes
- SHA256: `98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd`

## Allowed extracted source

- ZIP member: `data/musique_full_v1.0_train.jsonl`
- Path: `/home/hesong/AI-Agent-Projects/data/musique_official/train_only/musique_full_v1.0_train.jsonl`
- Size: `476696984` bytes
- Lines: `39876`
- SHA256: `b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034`

The train file was extracted with an exact ZIP-member allowlist. The official MuSiQue dev and test members have not been extracted or read. The archive is retained so its identity can be rechecked; downstream code must accept only the allowed train path until a later frozen protocol explicitly authorizes another split.

## Claim boundary

This code release does not redistribute the MuSiQue source rows or the derived
train/dev/shadow JSONL files. Anyone rebuilding or publishing derivatives must
retain attribution, link CC BY 4.0, identify modifications, and avoid implying
endorsement by the MuSiQue authors.
