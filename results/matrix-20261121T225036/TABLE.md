# Measured inference matrix

Effective prefill includes queueing and mixed decode work until the last request reaches its first token. Total decode is the sum of per-request token rates inside the common decode interval. Shared decode/request is the median rate in that same interval; per-request decode is the median of each request's own decode window and is defined even when no common interval exists. TTFT is the median time to first token.

Synthetic repeated notes with unique prefixes; 512-token forced output budget; prefix reuse excluded by a nonce prefix. This is not a quality test.

| Input tokens/request | C | Effective prefill tok/s | Total decode tok/s | Shared decode/request tok/s | Per-request decode tok/s | TTFT s | Shared decode s | Peak overlap | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 2057.7 | 94.1 | 94.1 | 98.4 | 0.2 | 5.4 | 1 | 1/1 |
| 512 | 2 | 430.1 | 131.1 | 65.6 | 69.7 | 2.4 | 7.2 | 2 | 2/2 |
| 512 | 4 | 2621.4 | 201.7 | 51.6 | 55.0 | 0.8 | 9.5 | 4 | 4/4 |
| 512 | 8 | 1432.5 | 280.7 | 32.8 | 36.7 | 1.9 | 11.3 | 8 | 8/8 |
| 2,048 | 1 | 2785.4 | 79.0 | 79.0 | 77.2 | 0.7 | 6.5 | 1 | 1/1 |
| 2,048 | 2 | 2526.5 | 126.6 | 63.3 | 62.3 | 1.3 | 7.8 | 2 | 2/2 |
| 2,048 | 4 | 2660.1 | 202.3 | 51.1 | 51.0 | 2.7 | 9.2 | 4 | 4/4 |
| 2,048 | 8 | 2229.8 | 348.4 | 43.2 | 43.4 | 4.3 | 10.5 | 8 | 8/8 |
| 8,192 | 1 | 2744.8 | 85.4 | 85.4 | 84.3 | 3.0 | 6.0 | 1 | 1/1 |
| 8,192 | 2 | 2703.9 | 111.5 | 55.7 | 63.2 | 4.6 | 7.1 | 2 | 2/2 |
| 8,192 | 4 | 2440.2 | 186.1 | 49.4 | 56.6 | 8.4 | 7.0 | 4 | 4/4 |
| 8,192 | 8 | 2521.3 | 291.5 | 37.3 | 38.7 | 14.4 | 9.2 | 8 | 8/8 |
| 32,768 | 1 | 2617.9 | 76.3 | 76.3 | 75.9 | 12.5 | 6.7 | 1 | 1/1 |
| 32,768 | 2 | 2447.9 | 123.5 | 61.7 | 63.6 | 19.8 | 6.9 | 2 | 2/2 |
| 32,768 | 4 | 2465.7 | 183.2 | 46.6 | 37.9 | 32.8 | 4.4 | 4 | 4/4 |
| 32,768 | 8 | 2461.5 | — | — | 12.1 | 58.9 | 0.0 | 6 | 8/8 |
| 131,072 | 1 | 1988.2 | 66.0 | 66.0 | 68.2 | 65.9 | 7.7 | 1 | 1/1 |
| 131,072 | 2 | 1936.2 | 94.1 | 47.1 | 38.5 | 100.7 | 2.8 | 2 | 2/2 |
| 131,072 | 4 | 1928.4 | — | — | 6.6 | 168.6 | 0.0 | 2 | 4/4 |
| 131,072 | 8 | 1915.5 | — | — | 5.4 | 305.1 | 0.0 | 3 | 8/8 |
| 200,000 | 1 | 1674.9 | 59.4 | 59.4 | 61.5 | 119.4 | 8.6 | 1 | 1/1 |
| 200,000 | 2 | 1642.4 | 71.3 | 35.6 | 31.0 | 181.9 | 1.7 | 2 | 2/2 |
| 200,000 | 4 | 1638.7 | — | — | 4.4 | 303.0 | 0.0 | 1 | 4/4 |
| 200,000 | 8 | 1634.3 | — | — | 4.0 | 549.6 | 0.0 | 2 | 8/8 |
| 400,000 | 1 | 1118.8 | 47.8 | 47.8 | 48.7 | 357.5 | 10.7 | 1 | 1/1 |
| 400,000 | 2 | 1106.9 | — | — | 25.3 | 540.7 | 0.0 | 1 | 2/2 |
| 400,000 | 4 | 1091.9 | — | — | 4.0 | 907.4 | 0.0 | 1 | 4/4 |
