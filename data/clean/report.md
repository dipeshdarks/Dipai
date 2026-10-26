# Data cleaning report

- documents in: **500**
- documents out: **479** (95.8% kept)
- estimated tokens: **521,443**

| stage | kept | dropped | modified | detail |
|---|---:|---:|---:|---|
| normalize | 500 | 0 | 38 | - |
| drop_empty | 500 | 0 | 0 | - |
| encoding_filter | 500 | 0 | 0 | - |
| length_filter | 500 | 0 | 0 | - |
| boilerplate | 480 | 20 | 0 | bullet_list=19, one_line_wall=1 |
| language | 479 | 1 | 0 | nl=1 |
| dedup_exact | 479 | 0 | 0 | - |
| dedup_near | 479 | 0 | 0 | - |
| quality | 479 | 0 | 0 | - |
| pii | 0 | 0 | 32 | - |

## PII detected
- credit_card: 5
- email: 21
- phone: 53

## Quality score histogram (deciles)
- 50-59: 3
- 60-69: 26
- 70-79: 450
