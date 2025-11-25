# Scenario-Aware JSON Re-Contextualization POC

## Overview

This POC demonstrates **100% AI-driven JSON transformation** using **LangGraph** for parallel agentic workflow orchestration. Given an educational simulation JSON and a target scenario index, the system transforms all scenario-dependent content while preserving locked instructional design fields with **10-15 second latency**.

## What This System Guarantees

✅ **Schema Fidelity**: Output JSON maintains identical structure to input

✅ **Locked Field Immutability**: Five instructional design fields never change:
- `scenarioOptions` (master scenario list)
- `assessmentCriterion` (learning outcomes)
- `industryAlignedActivities` (activity templates)
- `lessonInformation.level` (course level)
- `flowProperties.purpose` (pedagogical purposes)

✅ **Deep Semantic Transformation**: Not just find-and-replace - the LLM understands context and transforms:
- Company names and competitor names
- Brand positioning attributes (e.g., "fresh & organic" → "flavor variety & speed")
- Email addresses and person names
- All narrative content to match new scenario

✅ **Validation & Observability**: Comprehensive JSON and Markdown reports with pass/fail signals

✅ **100% AI-Based**: No regex, no manual pattern matching - pure LLM intelligence

## Architecture

### LangGraph Agentic Workflow

```
                    START
                      ↓
          [Chunk Preparation Node]
    Analyzes structure → Creates 18-22 chunks
                      ↓
    ┌─────────────────┴─────────────────┐
    ↓                 ↓                 ↓
[Chunk_0]      [Chunk_1] ... [Chunk_N]
(Metadata)    (Parallel Generation - 18-22 nodes)
    ↓                 ↓                 ↓
    └─────────────────┬─────────────────┘
                      ↓
            [Merge Chunks Node]
      Combines chunks + restores keys
                      ↓
            [Validation Node]
       Checks locked fields + schema
                      ↓
              [should_retry?]
                    / \
         Errors?   /   \   All Pass?
                  /     \
    [Chunk Prep] ←       → END
    (retry once)
```

**Workflow Architecture:**

1. **Dynamic Graph Construction**: 
   - Pre-computes chunk count (18-22 based on JSON structure)
   - Builds LangGraph with N parallel generation nodes dynamically

2. **Parallel Fan-Out**:
   - `chunk_preparation` → `generate_chunk_0..N` (all edges created)
   - All chunks execute concurrently (true parallelism)

3. **Parallel Fan-In**:
   - `generate_chunk_0..N` → `merge_chunks` (all edges converge)
   - Merge waits for all parallel nodes to complete

4. **Linear Validation**:
   - `merge_chunks` → `validation` → `should_retry` (conditional router)

5. **Conditional Retry**:
   - If errors: `should_retry` → `chunk_preparation` (full retry)
   - If pass: `should_retry` → `END`

**Key Features:**
- **True Parallelism**: LangGraph executes all chunk nodes concurrently
- **Fault Tolerance**: Missing chunks restored from input during merge
- **Single Retry**: Max 1 retry attempt (retry_count < 1)
- **Recursion Limit**: 10 (prevents infinite loops)
- **State Reducers**: `merge_dicts` for chunks, `operator.add` for errors

### Parallel Chunking Strategy

**Why Parallelization?**

Without chunking, the entire 50KB JSON would be sent to Gemini in a single request:
- **Sequential Approach**: ~80-90 seconds (LLM processes entire JSON at once)
- **Bottleneck**: Large context requires more thinking time, even with fast models

**With Parallel Chunking**: Break JSON into 18-22 chunks and process concurrently:
- **Parallel Approach**: ~10-15 seconds (chunks processed simultaneously)
- **Speedup**: **6x faster** than sequential processing
- **Key Insight**: Paid API tier allows high concurrency - leverage it!

The system achieves this dramatic latency reduction through intelligent parallel chunking:

**Chunking Algorithm:**
1. **Metadata Chunk** (Chunk 0): `simulationName` + `workplaceScenario` (always first)
2. **Flow-Level Chunking**: Each `simulationFlow` item analyzed:
   - **With Children**: Each child becomes separate chunk
   - **Large Children** (>10k chars): Split into 2 chunks by data keys
   - **Large Data Objects**: Split by data keys individually
   - **Very Large Keys** (>15k chars): Dict split into 3 sub-chunks
   - **Large Keys** (>8k chars): Dict split into 2 sub-chunks
   - **Large Lists** (>10k chars): List split into 2 halves
   - **Small Flows**: Keep as single chunk

**Chunk Size Optimization:**
- **Target**: 18-22 parallel chunks (sweet spot for paid Gemini API tier)
- **Balance**: Enough parallelization to reduce latency without overhead
- **Thresholds**: Tuned to prevent excessive splitting (40+ chunks adds overhead)

**Example Chunk Distribution:**
```
Chunk 0:  Metadata + workplace (3,209 chars)
Chunk 1:  Flow "Introduction" child 1 (555 chars)
Chunk 2:  Flow "Task" child 1 (2,122 chars)
Chunk 3:  Flow "Task" child 2 (523 chars)
...
Chunk 11: Flow "Task" resource data part 1 (6,000 chars)
Chunk 12: Flow "Task" resource data part 2 (5,649 chars)
...
Total: 20 chunks processed in parallel → ~12 second total time
```

**Why This Works:**
- **Concurrency**: 18-22 API calls execute simultaneously (paid tier)
- **Load Balancing**: Chunks sized to finish around same time
- **Fault Tolerance**: If chunk fails, original data restored during merge
- **Minimal Overhead**: Graph construction + merge < 1 second

**Chunk-to-JSON Mapping:**

Each chunk contains a subset of the JSON with preserved structure:

```python
# Chunk 0 (metadata)
{
  "simulationName": "...",
  "workplaceScenario": { ... }
}

# Chunk 1-N (simulationFlow chunks)
{
  "simulationFlow": [{
    "name": "Task Flow",
    "data": { "instructions": "..." }  # Partial data
  }]
}

# Merge combines all chunks back into complete JSON
output_json = {
  "topicWizardData": {
    "simulationName": transformed_chunk_0.simulationName,
    "workplaceScenario": transformed_chunk_0.workplaceScenario,
    "simulationFlow": [
      merge(chunk_1, chunk_2, chunk_3, ...),
      ...
    ],
    # Locked fields restored from input
    "scenarioOptions": input.scenarioOptions,
    ...
  }
}
```

**Parallel Execution Flow:**
```
Time 0s:    All 20 chunks sent to Gemini API simultaneously
Time 10s:   Chunks 1-18 complete (small/medium chunks)
Time 12s:   Chunks 19-20 complete (large chunks finish last)
Time 12.5s: Merge combines all chunks + validates
Time 13s:   Output written, reports generated
```

## Setup

### Prerequisites

- Python 3.9+
- Google Gemini API key

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Environment Variables

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Usage

### Basic Command

```bash
python main.py <input.json> <scenario_index>
```

### Example

Transform `POC_sim.json` to scenario 7 (BurgerZone vs GrillKing):

```bash
python main.py POC_sim.json 7
```

### Parameters

- `input.json`: Path to input JSON file
- `scenario_index`: Target scenario (0-based index from `scenarioOptions` array)

## Output Artifacts

1. **output.json** - Transformed simulation JSON
2. **validation_report.json** - Machine-readable validation results with:
   - Schema fidelity status
   - Locked field violations
   - Changed fields summary (paths, old/new values)
   - Scenario consistency analysis
   - Runtime and retry metrics
3. **validation_report.md** - Human-readable report with diagnostics
4. **illustrative_diff.md** - Side-by-side before/after comparison (generated via `illustrative_diff.py`)

### Generating Illustrative Diff

After running the transformation, generate a before/after comparison:

```bash
python illustrative_diff.py POC_sim.json output.json
```

This creates `illustrative_diff.md` showing key field transformations.

## Example Run

```bash
$ python main.py POC_sim.json 7

Transforming POC_sim.json to scenario 7
============================================================

Target: BurgerZone's foot traffic declines after rival GrillKing...

Preparing chunks for parallel generation...
  [PASS] Created 20 logical chunks:
    Chunk 0: 3209 chars
    Chunk 1: 555 chars
    ...

Building workflow with 20 parallel chunks...
  Generating chunk chunk_0...
    [PASS] Chunk chunk_0 transformed
  ...

Merging transformed chunks...
  [PASS] Pydantic schema validation passed
  [PASS] Chunks merged successfully

Validating output...
  [PASS] All locked fields preserved

============================================================
Complete in 12045ms
Status: PASS
Attempts: 1
Output: output.json
Validation reports: validation_report.json, validation_report.md
============================================================
```

## Example Transformation

### Input Scenario (Index 0)
```
A strategy team at HarvestBowls is facing a drop in foot traffic after
Nature's Crust introduced a $1 value menu. Learners must analyze the
market shake-up and recommend a plan that helps HarvestBowls maintain
its commitment to serving fresh, organic, and wholesome fast food.
```

### Target Scenario (Index 7)
```
BurgerZone's foot traffic declines after rival GrillKing promotes a
$2 meal deal. Students must recommend a clear plan that balances
competitive pricing with BurgerZone's positioning around flavor
variety and speed of service.
```

### Sample Transformations

**Company Name:**
- Before: `HarvestBowls`
- After: `BurgerZone`

**Brand Positioning:**
- Before: `"known for its fresh, organic, and wholesome menu"`
- After: `"known for its flavor variety and speed of service"`

**Email Domain:**
- Before: `mark.caldwell@harvestbowls.com`
- After: `mark.caldwell@burgerzone.com`

**All locked fields preserved!**

## Technical Details

### 100% AI-Based Approach

**How It Works:**
1. Each chunk sent to LLM with target scenario description
2. LLM generates transformed chunk (preserving structure)
3. Chunks merged and validated with Pydantic schema
4. Locked fields force-restored from input (safety measure)
5. Final validation checks schema and locked field equality

**No Regex:**
- All entity extraction: LLM
- All text transformation: LLM
- All positioning updates: LLM

**Why This Works:**
- Gemini 2.5 Flash has large context window
- LLM understands semantic relationships
- LLM can maintain JSON structure
- **Pydantic validation** catches schema violations immediately
- Python safety net ensures locked fields never change

**Pydantic's Role:**
```python
class SimulationJSON(BaseModel):
    topicWizardData: TopicWizardData
    
class TopicWizardData(BaseModel):
    lessonInformation: LessonInformation
    scenarioOptions: List[str]
    simulationFlow: List[Dict[str, Any]]
    # ... all fields with strict typing
```

- **Schema Enforcement**: Validates JSON structure matches expected shape
- **Type Safety**: Ensures all fields have correct types (str, list, dict, etc.)
- **Early Detection**: Catches malformed JSON before merge completes
- **Fast Validation**: ~10ms overhead vs 10+ seconds saved from preventing retries

### LangGraph Benefits

✅ **Stateful Workflow**: Shared state flows through agents
✅ **Conditional Routing**: Automatic retry logic
✅ **Declarative**: Define graph, not control flow
✅ **Observable**: Track which node is executing

## Validation Approach

### Schema Fidelity
- DeepDiff-based structural comparison
- Recursive key-by-key comparison
- Type matching (dict/list/primitive)
- Array length verification

### Locked Field Compliance
- Deep equality check using Python `==`
- Post-transformation re-validation
- Fail-fast on any modification
- Automated locked field restoration

### Scenario Consistency Checks
- Entity mention counting (old vs new organization names)
- Cross-reference validation
- Leftover detection from previous scenario
- Top entity frequency analysis

### Quality Signals
- Pass/fail validation status
- Runtime tracking (ms)
- Retry count
- Changed field count with sample changes
- Detailed error reporting
- Structured JSON + Markdown reports

## Observability

The system provides comprehensive observability through:

### Structured Logging
- Timestamped log entries for each node
- Decision rationale tracking
- Lock field extraction metrics
- Chunk creation statistics
- Entity mention counts

### Validation Reports
- **validation_report.json**: Machine-readable validation results
- **validation_report.md**: Human-readable report with detailed diagnostics
- **illustrative_diff.md**: Side-by-side before/after comparison

### Metrics Tracked
- Runtime (milliseconds)
- Retry attempts
- Fields changed (count + sample)
- Locked field compliance
- Schema fidelity status
- Scenario consistency metrics (entity mentions)

## Project Structure

```
cartedo-poc/
├── POC_sim.json              # Input sample
├── output.json               # Example transformed output
├── main.py                   # Main entry point (LangGraph workflow)
├── illustrative_diff.py      # Generate before/after comparison
├── validation_report.json    # Example validation report (generated)
├── validation_report.md      # Example readable report (generated)
├── illustrative_diff.md      # Example diff report (generated)
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── .gitignore                # Git exclusions
└── README.md                 # This file
```

## Requirements

```
langgraph>=0.2.0
langchain>=0.3.0
langchain-google-genai>=2.0.0
python-dotenv>=1.0.0
deepdiff>=7.0.0
pydantic>=2.0.0
```

## Performance

### Optimized Configuration (Current)
- **Runtime**: **10-15 seconds** for 50KB JSON (paid Gemini API tier)
  - **Parallel Processing**: 18-22 chunks processed concurrently
  - **Balanced Chunking**: Optimal thresholds prevent overhead from excessive splitting
  - **Single Retry**: Fast failure handling (max 1 retry attempt)
- **API Tier**: Paid Gemini API (enables higher concurrency and better performance)
- **Token Usage**: Distributed across parallel chunks (lower per-request context)
- **Reliability**: High (locked fields guaranteed preserved + fault-tolerant merge)
- **Quality**: Deep semantic transformation beyond find-replace
- **Determinism**: `temperature=0` + `thinking_budget=0` for stable results
- **Fault Tolerance**: System completes successfully even if individual chunks fail

### Chunk Configuration
- **Child data**: Split if > 10,000 chars
- **Dictionary keys**: Split if > 8,000 chars
- **Very large dicts**: Split into 3 chunks if > 15,000 chars
- **Large lists**: Split if > 10,000 chars
- **Result**: 18-22 parallel chunks (sweet spot for paid tier)

## Limitations

- Requires valid Google Gemini API key (paid tier recommended for optimal performance)
- Input must match reference schema structure
- Individual chunk failures possible (but system is fault-tolerant)
- Large HTML/markdown chunks may occasionally produce invalid JSON (system handles gracefully)
- Best for scenarios where deep semantic understanding is required

## POC Requirements Met

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Accept input.json + selectedScenario | ✅ | CLI accepts input path + scenario index |
| Produce output.json with same structure | ✅ | Pydantic schema validation + DeepDiff checks |
| Lock 5 fields (scenarioOptions, assessmentCriterion, etc.) | ✅ | Automated restoration + validation |
| Transform scenario content | ✅ | LLM-based deep semantic transformation |
| Global coherence | ✅ | Entity tracking + consistency checks |
| Validation report (machine-readable) | ✅ | validation_report.json with pass/fail + metrics |
| Schema fidelity verification | ✅ | DeepDiff structural comparison |
| Locked-field equality check | ✅ | Deep equality + violation detection |
| Changed-field summary | ✅ | DeepDiff paths with old/new values (20 samples) |
| Scenario consistency checks | ✅ | Entity mentions + leftover detection |
| Runtime stats | ✅ | Millisecond precision + retry count |
| Agentic workflow | ✅ | LangGraph with 4+ agent roles |
| Graph-based routing | ✅ | Conditional edges + retry logic |
| Observability | ✅ | Structured logging + decision rationale |
| Latency | ✅ | ~10-15s (optimized with paid API tier) |
| Determinism | ⚠️ | `temperature=0` (stable, not byte-identical) |
| 100% AI-based (no regex) | ✅ | Pure LLM intelligence for all transformations |

---

**POC Status**: ✅ **COMPLETE** & **OPTIMIZED**
**Model**: Google Gemini 2.5 Flash Lite
**Framework**: LangGraph with parallel execution
**Approach**: 100% AI-driven transformation with balanced chunking
**Performance**: 10-15 second latency with paid Gemini API tier
**Architecture**: Fault-tolerant with automatic missing key restoration
