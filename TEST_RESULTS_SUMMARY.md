# Test Results Summary: POC_sim.json - All Scenarios

**Test Date:** 2025-11-25
**Total Scenarios:** 37
**Command:** `python main.py POC_sim.json <scenario_index>`

---

## Overall Results

| Metric | Value |
|--------|-------|
| **Total Scenarios** | 37 |
| **Passed** | 37 |
| **Failed** | 0 |
| **Pass Rate** | 100% |

---

## Latency Statistics

| Statistic | Value (ms) | Value (s) |
|-----------|------------|-----------|
| **Average** | 6,936 ms | 6.9 s |
| **Minimum** | 6,085 ms | 6.1 s |
| **Maximum** | 8,873 ms | 8.9 s |
| **Total Time** | 256,614 ms | 256.6 s (4.3 min) |

---

## Detailed Results Per Scenario

| Scenario # | Status | Latency (ms) | Latency (s) |
|------------|--------|--------------|-------------|
| 0 | PASS | 6,218 | 6.2 |
| 1 | PASS | 7,509 | 7.5 |
| 2 | PASS | 6,752 | 6.8 |
| 3 | PASS | 6,779 | 6.8 |
| 4 | PASS | 7,143 | 7.1 |
| 5 | PASS | 6,862 | 6.9 |
| 6 | PASS | 6,534 | 6.5 |
| 7 | PASS | 6,850 | 6.9 |
| 8 | PASS | 6,334 | 6.3 |
| 9 | PASS | 7,682 | 7.7 |
| 10 | PASS | 6,341 | 6.3 |
| 11 | PASS | 7,680 | 7.7 |
| 12 | PASS | 7,113 | 7.1 |
| 13 | PASS | 6,756 | 6.8 |
| 14 | PASS | 7,633 | 7.6 |
| 15 | PASS | 7,220 | 7.2 |
| 16 | PASS | 6,185 | 6.2 |
| 17 | PASS | 7,004 | 7.0 |
| 18 | PASS | 6,794 | 6.8 |
| 19 | PASS | 6,687 | 6.7 |
| 20 | PASS | 6,736 | 6.7 |
| 21 | PASS | 7,117 | 7.1 |
| 22 | PASS | 6,337 | 6.3 |
| 23 | PASS | 7,963 | 8.0 |
| 24 | PASS | 8,873 | 8.9 |
| 25 | PASS | 8,352 | 8.4 |
| 26 | PASS | 6,085 | 6.1 |
| 27 | PASS | 6,624 | 6.6 |
| 28 | PASS | 6,538 | 6.5 |
| 29 | PASS | 7,013 | 7.0 |
| 30 | PASS | 6,657 | 6.7 |
| 31 | PASS | 6,294 | 6.3 |
| 32 | PASS | 6,987 | 7.0 |
| 33 | PASS | 6,967 | 7.0 |
| 34 | PASS | 6,517 | 6.5 |
| 35 | PASS | 6,671 | 6.7 |
| 36 | PASS | 7,099 | 7.1 |

---

## Performance Insights

- **Consistency:** All scenarios completed successfully with no failures
- **Performance Range:** Latency varied from 6.1s to 8.9s (2.8s difference)
- **Average Performance:** ~7 seconds per scenario transformation
- **Outliers:**
  - Fastest: Scenario 26 (6,085ms)
  - Slowest: Scenario 24 (8,873ms)

---

## Test Configuration

- **Model:** gemini-2.5-flash-lite
- **Workflow:** Parallel chunk generation with LangGraph
- **Validation:** Pydantic schema validation + locked field preservation
- **Retry Logic:** Up to 2 retry attempts on validation failures

---

## Conclusion

✅ **All 37 scenarios passed successfully**
✅ **100% pass rate achieved**
✅ **Average latency of ~7 seconds per scenario**
✅ **Consistent performance across all test cases**

The system demonstrates robust and reliable JSON transformation across all scenario variations with consistent validation and locked field preservation.
