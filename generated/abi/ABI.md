# Generated ABI (splicev-dev-synthetic, profile dev-synthetic)

Manifest SHA-256: `3cbe66ced87d74e9e0d651713d08607049605839c4c159db24236c042130438e`. Generated file; edit the manifest and run `python3 tools/generate_sources.py`.

## Unit descriptor (16 x 32-bit words, little-endian)

| field | word | lsb | width |
|---|---|---|---|
| magic | 0 | 0 | 32 |
| version | 1 | 0 | 16 |
| manifest_hash16 | 1 | 16 | 16 |
| frame_id | 2 | 0 | 32 |
| slot | 3 | 0 | 8 |
| opcode | 3 | 8 | 8 |
| phase | 3 | 16 | 8 |
| flags | 3 | 24 | 8 |
| layer | 4 | 0 | 16 |
| band | 4 | 16 | 8 |
| partition | 4 | 24 | 8 |
| worker | 5 | 0 | 8 |
| n_workers | 5 | 8 | 8 |
| lanes | 5 | 16 | 16 |
| dim_m | 6 | 0 | 32 |
| dim_n | 7 | 0 | 32 |
| dim_k | 8 | 0 | 32 |
| in_words | 9 | 0 | 32 |
| out_words | 10 | 0 | 32 |
| param0 | 11 | 0 | 32 |
| param1 | 12 | 0 | 32 |
| param2 | 13 | 0 | 32 |
| token | 14 | 0 | 32 |
| check | 15 | 0 | 32 |

## Unit response header (8 x 32-bit words)

| field | word | lsb | width |
|---|---|---|---|
| magic | 0 | 0 | 32 |
| frame_id | 1 | 0 | 32 |
| opcode | 2 | 0 | 8 |
| status | 2 | 8 | 8 |
| slot | 2 | 16 | 8 |
| worker | 2 | 24 | 8 |
| out_words | 3 | 0 | 32 |
| token | 4 | 0 | 32 |
| in_words | 5 | 0 | 32 |
| reserved | 6 | 0 | 32 |
| check | 7 | 0 | 32 |

## Unit response trailer (4 x 32-bit words, after the payload)

| field | word | lsb | width |
|---|---|---|---|
| sat_count | 0 | 0 | 32 |
| aux0 | 1 | 0 | 32 |
| aux1 | 2 | 0 | 32 |
| check | 3 | 0 | 32 |

Framing: every response is header (8 words) + out_words payload (even) + trailer (4 words); counters follow the payload because a worker streams its outputs as it computes them; the trailer check chains from the header check.

Check word: c = 0x9E3779B9; for w in words[0:n-1]: c = rotl32(c, 5) ^ w; the result is stored in the last word.

## Opcodes

- `CONV` = 0
- `GEMM` = 1
- `STENCIL` = 2
- `LUT` = 3
- `PERMUTE` = 4
- `SEGOPS` = 5
- `TAPMOD` = 6
- `HEAD` = 7
- `NOP` = 255

## Boundary packets

Each packet carries an 8-word header (magic, label, version, frame_id, generation, payload_bytes, count, check).

| label | dir | payload bytes | figure KB | fields |
|---|---|---|---|---|
| H0 | AIE->PL | 12800 | 12.8 | e:u8x6400, t:u8x6400 |
| H1 | AIE->PL | 49664 | 139.7 | e:u8x1856, t:u8x1856, xy:u16x1856, hyp_mu:i16x7424, hyp_prec:u16x7424, reserve_scores:u8x6400, rr_partner:u16x2048, rr_score:u8x2048 |
| P | AIE->PL | 19200 | 19.2 | scores:u8x19200 |
| H2 | PL->AIE | 20520 | 25.0 | ids_p3:u16x6400, ids_p4:u16x1600, ids_p5:u16x400, order:u16x1856, F:u32x1 |
| G | PL->AIE | 8192 | 8.2 | mask:u32x2048 |
| H3 | AIE->PL | 91648 | 100.0 | H:i8x16384, nu:i8x3072, need:i32x256, rho:i32x256, phi:i32x1024, n_partners:u16x256, evidence:i16x32768 |
| H3p | PL->AIE | 4096 | 4.1 | sources:u16x2048 |
| H3pp | AIE->PL | 106496 | 106.5 | tv:i8x98304, checks:i32x2048 |
| H4 | PL->AIE | 18432 | 20.0 | dV:i8x12288, boxes:i16x1024, xi:i8x4096 |
| H5 | AIE->PL | 96008 | 100.0 | count:u32x1, cand:u32x24000 |

Generated payload total: 427056 bytes (427.1 decimal KB); figure labels total 535.5 KB.
Figure labels are rounded display values; DMA lengths, FIFO sizes and counters use the generated sizes (roadmap 2.3).
