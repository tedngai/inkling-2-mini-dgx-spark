# Measured inference matrix

Effective prefill includes queueing and mixed decode work until the last request reaches its first token. Total decode is the sum of per-request token rates inside the common decode interval. Decode/request is the median rate in that same interval. TTFT is the median time to first token.

Synthetic repeated notes with unique prefixes; 512-token forced output budget; prefix reuse excluded by a nonce prefix. This is not a quality test.

| Input tokens/request | C | Effective prefill tok/s | Total decode tok/s | Decode/request tok/s | TTFT s | Shared decode s | Peak overlap | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 2057.7 | 94.1 | 94.1 | 0.2 | 5.4 | 1 | 1/1 |
| 512 | 2 | 430.1 | 131.1 | 65.6 | 2.4 | 7.2 | 2 | 2/2 |
| 512 | 4 | 2621.4 | 201.7 | 51.6 | 0.8 | 9.5 | 4 | 4/4 |
| 512 | 8 | 1432.5 | 280.7 | 32.8 | 1.9 | 11.3 | 8 | 8/8 |
| 2,048 | 1 | 2785.4 | 79.0 | 79.0 | 0.7 | 6.5 | 1 | 1/1 |
| 2,048 | 2 | 2526.5 | 126.6 | 63.3 | 1.3 | 7.8 | 2 | 2/2 |
| 2,048 | 4 | 2660.1 | 202.3 | 51.1 | 2.7 | 9.2 | 4 | 4/4 |
| 2,048 | 8 | 2229.8 | 348.4 | 43.2 | 4.3 | 10.5 | 8 | 8/8 |
| 8,192 | 1 | 2744.8 | 85.4 | 85.4 | 3.0 | 6.0 | 1 | 1/1 |
| 8,192 | 2 | 2703.9 | 111.5 | 55.7 | 4.6 | 7.1 | 2 | 2/2 |
