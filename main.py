"""
LangGraph Workflow for Scenario-Aware JSON Re-Contextualization
Agentic architecture using 100% AI-based transformation with Pydantic validation
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Annotated
from pydantic import BaseModel, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from deepdiff import DeepDiff
import operator

load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Reducer function to merge transformed chunks from parallel nodes
def merge_dicts(existing: dict, new: dict) -> dict:
    """Merge dictionaries for parallel chunk processing"""
    return {**existing, **new}


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class LessonInformation(BaseModel):
    """Schema for lesson information"""
    level: str
    lesson: str


class TopicWizardData(BaseModel):
    """Schema for the main data structure"""
    lessonInformation: LessonInformation
    scenarioOptions: List[str]
    selectedScenarioOption: str
    assessmentCriterion: List[Dict[str, Any]]
    selectedAssessmentCriterion: List[Dict[str, Any]]
    simulationName: str
    workplaceScenario: Dict[str, Any]
    simulationFlow: List[Dict[str, Any]]
    industryAlignedActivities: List[Dict[str, Any]]
    selectedIndustryAlignedActivities: List[Dict[str, Any]]


class SimulationJSON(BaseModel):
    """Root schema for the simulation JSON"""
    topicWizardData: TopicWizardData


# ============================================================================
# HELPERS
# ============================================================================

def extract_purposes(json_data: dict) -> List[str]:
    """Extract all flowProperties.purpose fields from simulationFlow"""
    purposes = []
    for flow in json_data["topicWizardData"]["simulationFlow"]:
        if "flowProperties" in flow and "purpose" in flow["flowProperties"]:
            purposes.append(flow["flowProperties"]["purpose"])
        if "children" in flow:
            for child in flow["children"]:
                if "flowProperties" in child and "purpose" in child["flowProperties"]:
                    purposes.append(child["flowProperties"]["purpose"])
    return purposes


def extract_entity_mentions(json_data: dict) -> Dict[str, int]:
    """Extract and count entity mentions (company names, people, etc.) from JSON, excluding locked flowProperties"""
    # Create a copy without locked flowProperties for counting
    json_copy = json.loads(json.dumps(json_data))

    # Remove locked flowProperties from all flows for entity counting
    if "topicWizardData" in json_copy and "simulationFlow" in json_copy["topicWizardData"]:
        for flow in json_copy["topicWizardData"]["simulationFlow"]:
            if "flowProperties" in flow:
                del flow["flowProperties"]
            if "children" in flow:
                for child in flow["children"]:
                    if "flowProperties" in child:
                        del child["flowProperties"]

    json_str = json.dumps(json_copy).lower()

    # Common entity patterns to track
    entities = {}

    # Extract company/brand names from scenario text
    scenario = json_data.get("topicWizardData", {}).get("selectedScenarioOption", "")
    words = scenario.split()

    # Count capitalized words (likely proper nouns)
    for word in words:
        if word and word[0].isupper() and len(word) > 3:
            clean_word = word.strip(".,!?;:")
            if clean_word:
                entities[clean_word] = json_str.count(clean_word.lower())

    # Extract organization name
    org_name = json_data.get("topicWizardData", {}).get("workplaceScenario", {}).get("background", {}).get("organizationName", "")
    if org_name:
        entities[org_name] = json_str.count(org_name.lower())

    return entities


def check_scenario_consistency(input_json: dict, output_json: dict, target_scenario: str) -> Dict[str, Any]:
    """Check if output is globally coherent with the target scenario"""
    logger.info("Running scenario consistency checks (excluding locked flowProperties)...")

    input_entities = extract_entity_mentions(input_json)
    output_entities = extract_entity_mentions(output_json)

    # Check for leftover entities from old scenario
    old_org = input_json["topicWizardData"]["workplaceScenario"]["background"]["organizationName"]
    new_org = output_json["topicWizardData"]["workplaceScenario"]["background"]["organizationName"]

    # Use counts from extract_entity_mentions which excludes locked flowProperties
    old_org_count = output_entities.get(old_org, 0)
    new_org_count = output_entities.get(new_org, 0)

    logger.info(f"Entity counts (excluding locked variables): '{new_org}' = {new_org_count}, '{old_org}' = {old_org_count}")

    issues = []
    if old_org_count > 0 and old_org.lower() != new_org.lower():
        issues.append(f"Old organization '{old_org}' still appears {old_org_count} times")

    consistency_report = {
        "old_organization": old_org,
        "new_organization": new_org,
        "new_organization_mentions": new_org_count,
        "old_organization_leftover_mentions": old_org_count,
        "top_entities_in_output": dict(sorted(output_entities.items(), key=lambda x: x[1], reverse=True)[:10]),
        "issues": issues,
        "status": "PASS" if len(issues) == 0 else "FAIL"
    }

    logger.info(f"Consistency check: {consistency_report['status']} - New org '{new_org}' appears {new_org_count} times, old org '{old_org}' appears {old_org_count} times")

    return consistency_report


def generate_validation_reports(state: dict, runtime_ms: float):
    """Generate comprehensive validation reports in JSON and Markdown formats"""
    logger.info("Generating validation reports...")

    input_json = state["input_json"]
    output_json = state["output_json"]
    locked = state["locked_fields"]

    # Schema fidelity check using DeepDiff
    diff = DeepDiff(input_json, output_json, ignore_order=False, view='tree')

    # Extract changed fields
    changed_fields = []
    if 'values_changed' in diff:
        for change in diff['values_changed']:
            path = str(change.path())
            old_val = str(change.t1)[:100]  # Truncate to 100 chars
            new_val = str(change.t2)[:100]
            changed_fields.append({
                "path": path,
                "type": "modified",
                "old_value": old_val,
                "new_value": new_val
            })

    # Check locked fields violations
    violations = []
    if output_json["topicWizardData"]["scenarioOptions"] != locked["scenarioOptions"]:
        violations.append("scenarioOptions modified")
    if output_json["topicWizardData"]["assessmentCriterion"] != locked["assessmentCriterion"]:
        violations.append("assessmentCriterion modified")
    if output_json["topicWizardData"]["industryAlignedActivities"] != locked["industryAlignedActivities"]:
        violations.append("industryAlignedActivities modified")
    if output_json["topicWizardData"]["lessonInformation"]["level"] != locked["level"]:
        violations.append("lessonInformation.level modified")

    output_purposes = extract_purposes(output_json)
    if output_purposes != locked["flowProperties_purposes"]:
        violations.append("flowProperties.purpose modified")

    # Scenario consistency checks
    consistency = check_scenario_consistency(input_json, output_json, state["target_scenario"])

    # Build JSON report
    # Schema fidelity: PASS if only values changed or array items added/removed (dictionary keys must stay the same)
    # Allow: values_changed, repetition_change, iterable_item_added, iterable_item_removed
    # Fail on: dictionary_item_added, dictionary_item_removed, type_changes
    allowed_changes = ['values_changed', 'repetition_change', 'iterable_item_added', 'iterable_item_removed']
    schema_ok = (len(diff) == 0 or all(k in allowed_changes for k in diff.keys()))

    json_report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "schema_fidelity": "PASS" if schema_ok else "FAIL",
            "locked_fields": "PASS" if len(violations) == 0 else "FAIL",
            "validation_errors": len(state["errors"]),
            "consistency_issues": len(consistency["issues"]),
            "runtime_ms": runtime_ms,
            "retry_count": state["retry_count"],
            "overall_status": "PASS" if len(violations) == 0 and len(state["errors"]) == 0 and len(consistency["issues"]) == 0 else "FAIL"
        },
        "details": {
            "schema_issues": [],
            "locked_field_violations": violations,
            "validation_errors": state["errors"],
            "consistency_issues": consistency["issues"],
            "changed_fields_count": len(changed_fields),
            "changed_fields_sample": changed_fields[:20]  # First 20 changes
        },
        "locked_fields": {
            "scenarioOptions_count": len(locked["scenarioOptions"]),
            "assessmentCriterion_count": len(locked["assessmentCriterion"]),
            "industryAlignedActivities_count": len(locked["industryAlignedActivities"]),
            "level": locked["level"],
            "flowProperties_purposes_count": len(locked["flowProperties_purposes"])
        },
        "scenario_consistency": consistency
    }

    # Save JSON report
    with open("validation_report.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    # Generate Markdown report
    md_lines = [
        "# Validation Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"## Overall Status: {'✅ PASS' if json_report['summary']['overall_status'] == 'PASS' else '❌ FAIL'}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Status | Details |",
        "|--------|--------|---------|",
        f"| **Schema Fidelity** | {'✅ PASS' if json_report['summary']['schema_fidelity'] == 'PASS' else '❌ FAIL'} | {len(json_report['details']['schema_issues'])} issues |",
        f"| **Locked Fields** | {'✅ PASS' if json_report['summary']['locked_fields'] == 'PASS' else '❌ FAIL'} | {len(violations)} violations |",
        f"| **Validation Errors** | {'✅ PASS' if json_report['summary']['validation_errors'] == 0 else '❌ FAIL'} | {json_report['summary']['validation_errors']} errors |",
        f"| **Consistency Issues** | {'✅ PASS' if json_report['summary']['consistency_issues'] == 0 else '❌ FAIL'} | {json_report['summary']['consistency_issues']} issues |",
        f"| **Runtime** | {runtime_ms:.2f} ms | - |",
        f"| **Retry Count** | {state['retry_count']} | - |",
        f"| **Fields Changed** | {len(changed_fields)} | - |",
        "",
        "---",
        "",
        "## Schema Fidelity",
        "",
    ]

    if json_report['summary']['schema_fidelity'] == 'PASS':
        md_lines.append("✅ **Output JSON has identical schema structure to input.**")
    else:
        md_lines.append("❌ **Schema structure violations detected:**")
        for issue in json_report['details']['schema_issues']:
            md_lines.append(f"- {issue}")

    md_lines.extend([
        "",
        "---",
        "",
        "## Locked Fields Compliance",
        ""
    ])

    if len(violations) == 0:
        md_lines.append("✅ **All locked fields remain unchanged.**")
    else:
        md_lines.append("❌ **Locked field violations detected:**")
        for v in violations:
            md_lines.append(f"- {v}")

    md_lines.extend([
        "",
        "**Locked Fields Summary:**",
        f"- `scenarioOptions`: {json_report['locked_fields']['scenarioOptions_count']} scenarios",
        f"- `assessmentCriterion`: {json_report['locked_fields']['assessmentCriterion_count']} criteria",
        f"- `industryAlignedActivities`: {json_report['locked_fields']['industryAlignedActivities_count']} activities",
        f"- `lessonInformation.level`: {json_report['locked_fields']['level']}",
        f"- `flowProperties.purpose`: {json_report['locked_fields']['flowProperties_purposes_count']} purposes",
        "",
        "---",
        "",
        "## Scenario Consistency Checks",
        ""
    ])

    if consistency["status"] == "PASS":
        md_lines.append("✅ **Global scenario coherence verified:**")
        md_lines.append(f"- Old organization `{consistency['old_organization']}`: {consistency['old_organization_leftover_mentions']} mentions (expected: 0)")
        md_lines.append(f"- New organization `{consistency['new_organization']}`: {consistency['new_organization_mentions']} mentions")
        md_lines.append("")
        md_lines.append("**Top entities in output:**")
        for entity, count in list(consistency['top_entities_in_output'].items())[:5]:
            md_lines.append(f"- `{entity}`: {count} mentions")
    else:
        md_lines.append("❌ **Consistency issues detected:**")
        for issue in consistency["issues"]:
            md_lines.append(f"- {issue}")

    md_lines.extend([
        "",
        "---",
        "",
        "## Validation Errors",
        ""
    ])

    if len(state["errors"]) == 0:
        md_lines.append("✅ **No validation errors.**")
    else:
        md_lines.append(f"❌ **{len(state['errors'])} validation errors detected:**")
        for err in state["errors"]:
            md_lines.append(f"- {err}")

    md_lines.extend([
        "",
        "---",
        "",
        "## Changed Fields Summary",
        "",
        f"**Total fields changed:** {len(changed_fields)}",
        "",
        "**Sample of changes (first 10):**",
        ""
    ])

    for change in changed_fields[:10]:
        md_lines.append(f"- **{change['path']}** ({change['type']})")
        md_lines.append(f"  - Old: `{change['old_value']}`")
        md_lines.append(f"  - New: `{change['new_value']}`")

    if len(changed_fields) > 10:
        md_lines.append(f"\n... and {len(changed_fields) - 10} more changes.")

    md_lines.extend([
        "",
        "---",
        "",
        "**End of Report**",
        ""
    ])

    # Save Markdown report
    with open("validation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("✅ Validation reports saved: validation_report.json, validation_report.md")

    return json_report


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries"""
    return {**left, **right}

class State(TypedDict):
    """Workflow state"""
    input_json: dict
    target_scenario: str
    locked_fields: dict
    learning_context: dict
    transformable_parts: dict
    chunks: dict  # Store chunks for parallel processing
    transformed_chunks: Annotated[dict, merge_dicts]  # Store results from parallel nodes
    output_json: dict
    errors: Annotated[list, operator.add]  # Reducer to concatenate error lists from parallel nodes
    retry_count: int
    start_time: float
    validation_report: dict  # Store validation results


def chunk_preparation_node(state: State) -> State:
    """Split into logical chunks based on structure"""
    logger.info("NODE: chunk_preparation - Splitting JSON into logical chunks for parallel processing")
    print("Preparing chunks for parallel generation...")

    simulation_flow = state["transformable_parts"]["simulationFlow"]

    # Analyze structure to determine optimal chunks
    chunks = {}
    chunk_idx = 0

    # Chunk 0: Metadata + workplaceScenario
    chunks[f"chunk_{chunk_idx}"] = {
        "simulationName": state["transformable_parts"]["simulationName"],
        "workplaceScenario": state["transformable_parts"]["workplaceScenario"]
    }
    chunk_idx += 1

    # Process each simulationFlow item
    for flow in simulation_flow:
        # If flow has children, split them into separate chunks
        if "children" in flow and isinstance(flow["children"], list) and len(flow["children"]) > 0:
            children = flow["children"]
            # Each child becomes its own chunk, split large children further
            for child in children:
                # Check if child has large data that should be split
                if "data" in child and isinstance(child.get("data"), dict):
                    child_data_size = len(json.dumps(child["data"]))
                    if child_data_size > 10000:
                        # Split large child into 2 chunks
                        print(f"  Large child detected ({child_data_size} chars), splitting into 2 chunks...")
                        child_data_keys = list(child["data"].keys())
                        mid_point = len(child_data_keys) // 2

                        # First half
                        flow_copy1 = json.loads(json.dumps(flow))
                        child_copy1 = json.loads(json.dumps(child))
                        child_copy1["data"] = {k: child["data"][k] for k in child_data_keys[:mid_point]}
                        flow_copy1["children"] = [child_copy1]
                        chunks[f"chunk_{chunk_idx}"] = {"simulationFlow": [flow_copy1]}
                        chunk_idx += 1

                        # Second half
                        flow_copy2 = json.loads(json.dumps(flow))
                        child_copy2 = json.loads(json.dumps(child))
                        child_copy2["data"] = {k: child["data"][k] for k in child_data_keys[mid_point:]}
                        flow_copy2["children"] = [child_copy2]
                        chunks[f"chunk_{chunk_idx}"] = {"simulationFlow": [flow_copy2]}
                        chunk_idx += 1
                    else:
                        # Normal child - one chunk
                        flow_copy = json.loads(json.dumps(flow))
                        flow_copy["children"] = [child]
                        chunks[f"chunk_{chunk_idx}"] = {"simulationFlow": [flow_copy]}
                        chunk_idx += 1
                else:
                    # Child without data - one chunk
                    flow_copy = json.loads(json.dumps(flow))
                    flow_copy["children"] = [child]
                    chunks[f"chunk_{chunk_idx}"] = {"simulationFlow": [flow_copy]}
                    chunk_idx += 1
        # If flow has large data object, split by data keys (and split large keys further)
        elif "data" in flow and isinstance(flow["data"], dict) and len(flow["data"]) > 5:
            # Large data object - split into separate chunks by key
            data_keys = list(flow["data"].keys())
            print(f"  Large flow detected with {len(data_keys)} data keys, splitting...")
            for key in data_keys:
                flow_copy = json.loads(json.dumps(flow))
                key_data = flow["data"][key]
                key_data_size = len(json.dumps(key_data))

                # If this single key's data is large, split it (regardless of type)
                if key_data_size > 8000:
                    if isinstance(key_data, dict):
                        # Dict: split by keys into multiple chunks
                        sub_keys = list(key_data.keys())

                        # Determine split count based on size
                        if key_data_size > 15000:
                            num_splits = 3  # Very large -> 3 chunks
                            print(f"    Key '{key}' is very large dict ({key_data_size} chars), splitting into 3 chunks...")
                        else:
                            num_splits = 2  # Large -> 2 chunks
                            print(f"    Key '{key}' is large dict ({key_data_size} chars), splitting into 2 chunks...")

                        chunk_size = len(sub_keys) // num_splits
                        for i in range(num_splits):
                            start_idx = i * chunk_size
                            end_idx = (i + 1) * chunk_size if i < num_splits - 1 else len(sub_keys)

                            flow_copy_split = json.loads(json.dumps(flow))
                            flow_copy_split["data"] = {key: {k: key_data[k] for k in sub_keys[start_idx:end_idx]}}
                            chunks[f"chunk_{chunk_idx}"] = {"simulationFlow": [flow_copy_split]}
                            chunk_idx += 1
                    elif isinstance(key_data, list) and key_data_size > 10000:
                        # List: only split if very large
                        print(f"    Key '{key}' is very large list ({key_data_size} chars), splitting into 2 chunks...")
                        mid = len(key_data) // 2

                        flow_copy1 = json.loads(json.dumps(flow))
                        flow_copy1["data"] = {key: key_data[:mid]}
                        chunks[f"chunk_{chunk_idx}"] = {"simulationFlow": [flow_copy1]}
                        chunk_idx += 1

                        flow_copy2 = json.loads(json.dumps(flow))
                        flow_copy2["data"] = {key: key_data[mid:]}
                        chunks[f"chunk_{chunk_idx}"] = {"simulationFlow": [flow_copy2]}
                        chunk_idx += 1
                    else:
                        # String or other, or list under 10k: keep as single chunk
                        flow_copy["data"] = {key: key_data}
                        chunks[f"chunk_{chunk_idx}"] = {"simulationFlow": [flow_copy]}
                        chunk_idx += 1
                else:
                    # Normal size key - one chunk
                    flow_copy["data"] = {key: key_data}
                    chunks[f"chunk_{chunk_idx}"] = {"simulationFlow": [flow_copy]}
                    chunk_idx += 1
        else:
            # Small/medium flow - entire flow is one chunk
            chunks[f"chunk_{chunk_idx}"] = {"simulationFlow": [flow]}
            chunk_idx += 1

    NUM_CHUNKS = chunk_idx
    sizes = [len(json.dumps(chunks[f"chunk_{i}"])) for i in range(NUM_CHUNKS)]
    logger.info(f"Created {NUM_CHUNKS} logical chunks for parallel generation (total size: {sum(sizes)} chars)")
    print(f"  [PASS] Created {NUM_CHUNKS} logical chunks:")
    for i in range(NUM_CHUNKS):
        print(f"    Chunk {i}: {sizes[i]} chars")

    state["chunks"] = chunks

    return state


def generate_chunk_node(chunk_id: str):
    """Factory function to create chunk-specific generation nodes"""

    def node_fn(state: State) -> State:
        """Generate transformed JSON for a specific chunk"""
        print(f"  Generating chunk {chunk_id}...")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            thinking_budget=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        chunk_data = state["chunks"][chunk_id]

        prompt = f"""Transform this JSON chunk to match the new scenario. Keep exact same structure.

TARGET SCENARIO:
{state["target_scenario"]}

CRITICAL STRUCTURE PRESERVATION RULES:
1. Update ALL company names, competitor names, people names, emails, descriptions to match new scenario
2. Keep EXACT same dictionary/object keys - DO NOT add or remove any dictionary keys
3. Arrays/lists can have items added or removed to fit the new scenario context (e.g., adjust number of questions, resources)
4. Preserve all field names and object structures - only modify values and array contents
5. Adapt brand positioning, metrics, and narrative details to fit the new industry context

CHUNK TO TRANSFORM:
{json.dumps(chunk_data, indent=2)}

OUTPUT: Complete transformed JSON with same dictionary keys (array lengths can change if needed)."""

        try:
            response = llm.invoke([
                SystemMessage(content="Transform JSON precisely. Output only valid JSON."),
                HumanMessage(content=prompt)
            ])

            content = response.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            transformed_chunk = json.loads(content)

            print(f"    [PASS] Chunk {chunk_id} transformed")

            # Return only the transformed_chunks update (reducer will merge)
            return {"transformed_chunks": {chunk_id: transformed_chunk}}

        except Exception as e:
            print(f"    [FAIL] Chunk {chunk_id} failed: {e}")
            return {"errors": [f"chunk_{chunk_id}_failed: {str(e)}"], "retry_count": 1}

    return node_fn


def merge_chunks_node(state: State) -> State:
    """Merge transformed chunks back into complete output"""
    logger.info(f"NODE: merge_chunks - Merging {len(state['transformed_chunks'])} transformed chunks into output JSON")
    print("Merging transformed chunks...")

    try:
        # Build complete output by merging chunks
        output_json = json.loads(json.dumps(state["input_json"]))  # Deep copy

        # Chunk 0 has metadata
        if "chunk_0" in state["transformed_chunks"]:
            chunk_0 = state["transformed_chunks"]["chunk_0"]
            if "simulationName" in chunk_0:
                output_json["topicWizardData"]["simulationName"] = chunk_0["simulationName"]
            if "workplaceScenario" in chunk_0:
                output_json["topicWizardData"]["workplaceScenario"] = chunk_0["workplaceScenario"]

        # Update simulationFlow in-place - only modify transformed content, keep locked fields
        # Group transformed chunks by flow name
        transformed_by_flow = {}  # flow_name -> {children: [], data: {}}

        for chunk_key in sorted(state["transformed_chunks"].keys()):
            if chunk_key == "chunk_0":
                continue  # Already processed metadata

            chunk = state["transformed_chunks"][chunk_key]
            if "simulationFlow" not in chunk or len(chunk["simulationFlow"]) == 0:
                continue

            for flow_item in chunk["simulationFlow"]:
                flow_name = flow_item.get("name", "unknown")

                if flow_name not in transformed_by_flow:
                    transformed_by_flow[flow_name] = {"children": [], "data": {}}

                # Collect transformed children
                if "children" in flow_item and flow_item["children"]:
                    transformed_by_flow[flow_name]["children"].extend(flow_item["children"])

                # Collect transformed data keys
                if "data" in flow_item and flow_item["data"]:
                    transformed_by_flow[flow_name]["data"].update(flow_item["data"])

        # Update output_json simulationFlow in-place (DON'T overwrite, just update content)
        for flow in output_json["topicWizardData"]["simulationFlow"]:
            flow_name = flow.get("name")
            if flow_name in transformed_by_flow:
                # Update children content (keep original flowProperties untouched)
                if transformed_by_flow[flow_name]["children"]:
                    for i, transformed_child in enumerate(transformed_by_flow[flow_name]["children"]):
                        if i < len(flow.get("children", [])):
                            # Update child content but keep original flowProperties
                            original_child = flow["children"][i]
                            for key, value in transformed_child.items():
                                if key != "flowProperties":  # NEVER overwrite flowProperties
                                    original_child[key] = value

                # Update data content
                if transformed_by_flow[flow_name]["data"]:
                    if "data" not in flow:
                        flow["data"] = {}
                    flow["data"].update(transformed_by_flow[flow_name]["data"])
        output_json["topicWizardData"]["selectedScenarioOption"] = state["target_scenario"]

        # Restore any missing dictionary keys from input (preserve structure)
        # This fixes cases where chunking caused keys to be dropped
        def restore_missing_keys(input_obj, output_obj):
            """Recursively restore any dictionary keys that exist in input but not in output"""
            if isinstance(input_obj, dict) and isinstance(output_obj, dict):
                for key in input_obj:
                    if key not in output_obj:
                        # Key was removed - restore it with original value
                        output_obj[key] = input_obj[key]
                    elif isinstance(input_obj[key], dict) and isinstance(output_obj[key], dict):
                        # Recurse into nested dicts
                        restore_missing_keys(input_obj[key], output_obj[key])
                    elif isinstance(input_obj[key], list) and isinstance(output_obj[key], list):
                        # For lists, recurse into each item if they're dicts
                        for i in range(min(len(input_obj[key]), len(output_obj[key]))):
                            if isinstance(input_obj[key][i], dict) and isinstance(output_obj[key][i], dict):
                                restore_missing_keys(input_obj[key][i], output_obj[key][i])
        
        restore_missing_keys(state["input_json"], output_json)

        # Validate output structure with Pydantic (preserve key order)
        try:
            validated = SimulationJSON(**output_json)
            print("  [PASS] Pydantic schema validation passed")
            # Don't use model_dump() - it reorders keys based on model definition
        except ValidationError as ve:
            print(f"  [FAIL] Pydantic validation failed: {ve.error_count()} errors")
            for error in ve.errors()[:3]:
                print(f"    - {error['loc']}: {error['msg']}")
            state["errors"].append("pydantic_validation_failed")
            state["retry_count"] += 1
            return state

        # Locked fields already in output_json from deep copy (line 337) - no restoration needed!

        state["output_json"] = output_json
        state["errors"] = []
        print("  [PASS] Chunks merged successfully")

    except Exception as e:
        print(f"  [FAIL] Merge failed: {e}")
        state["errors"].append(f"merge_failed: {str(e)}")
        state["retry_count"] += 1

    return state


def validation_node(state: State) -> State:
    """Validate output - checks locked fields and schema compliance"""
    logger.info("NODE: validation - Checking locked fields and schema compliance")
    print("Validating output...")

    # Check if we have output
    if not state["output_json"]:
        print("  [FAIL] No output JSON generated")
        return state

    # Check locked fields
    violations = []
    output = state["output_json"]
    locked = state["locked_fields"]

    if output["topicWizardData"]["scenarioOptions"] != locked["scenarioOptions"]:
        violations.append("scenarioOptions modified")

    if output["topicWizardData"]["assessmentCriterion"] != locked["assessmentCriterion"]:
        violations.append("assessmentCriterion modified")

    if output["topicWizardData"]["industryAlignedActivities"] != locked["industryAlignedActivities"]:
        violations.append("industryAlignedActivities modified")

    if output["topicWizardData"]["lessonInformation"]["level"] != locked["level"]:
        violations.append("lessonInformation.level modified")

    output_purposes = extract_purposes(output)
    if output_purposes != locked["flowProperties_purposes"]:
        violations.append("flowProperties.purpose modified")

    if violations:
        print(f"  [FAIL] {len(violations)} locked field violations")
        for v in violations:
            print(f"    - {v}")
        state["errors"].extend(violations)
        state["retry_count"] += 1
    else:
        print("  [PASS] All locked fields preserved")
        state["errors"] = []

    return state


def should_retry(state: State) -> str:
    """Decide if we should retry generation"""
    # Only retry if we have errors AND haven't exceeded retry limit
    # retry_count starts at 0, increments after each attempt
    # So retry_count=0 means first attempt, retry_count=1 means one retry done
    if state["errors"] and state["retry_count"] < 1:
        print(f"  Retrying generation (attempt {state['retry_count'] + 1})...")
        return "generation"
    elif state["errors"]:
        print(f"  Max retries reached ({state['retry_count']} attempts), stopping with errors")
    return "end"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run(input_path: str, scenario_index: int, output_path: str = "output.json"):
    """Main entry point"""
    print(f"\nTransforming {input_path} to scenario {scenario_index}")
    print("="*60)

    # Load input
    with open(input_path, 'r', encoding='utf-8') as f:
        input_json = json.load(f)

    target_scenario = input_json["topicWizardData"]["scenarioOptions"][scenario_index]

    print(f"\nTarget: {target_scenario[:100]}...\n")

    # Build and run workflow
    start = time.time()

    # First pass: create chunks to determine actual count
    transformable_parts_temp = {
        "simulationName": input_json["topicWizardData"]["simulationName"],
        "selectedScenarioOption": input_json["topicWizardData"]["selectedScenarioOption"],
        "workplaceScenario": input_json["topicWizardData"]["workplaceScenario"],
        "simulationFlow": input_json["topicWizardData"]["simulationFlow"]
    }
    
    # Quick chunk estimation to build workflow
    temp_state = {"transformable_parts": transformable_parts_temp, "chunks": {}}
    chunk_preparation_node(temp_state)
    num_chunks = len(temp_state["chunks"])

    print(f"Building workflow with {num_chunks} parallel chunks...")

    workflow = StateGraph(State)
    workflow.add_node("chunk_preparation", chunk_preparation_node)

    # Add N chunk generation nodes dynamically based on actual chunk count
    for i in range(num_chunks):
        workflow.add_node(f"generate_chunk_{i}", generate_chunk_node(f"chunk_{i}"))

    workflow.add_node("merge_chunks", merge_chunks_node)
    workflow.add_node("validation", validation_node)

    # Set entry point
    workflow.set_entry_point("chunk_preparation")

    # Parallel generation: all chunks run concurrently after preparation
    for i in range(num_chunks):
        workflow.add_edge("chunk_preparation", f"generate_chunk_{i}")

    # All chunks must complete before merge
    for i in range(num_chunks):
        workflow.add_edge(f"generate_chunk_{i}", "merge_chunks")

    # Merge then validate
    workflow.add_edge("merge_chunks", "validation")

    # Validation with retry logic
    workflow.add_conditional_edges("validation", should_retry, {
        "generation": "chunk_preparation",  # Retry from chunking
        "end": END
    })

    app = workflow.compile()
    
    # Configure recursion limit for faster execution
    app.recursion_limit = 10

    # Initialize state with minimal preprocessing inline
    locked_fields = {
        "scenarioOptions": input_json["topicWizardData"]["scenarioOptions"],
        "assessmentCriterion": input_json["topicWizardData"]["assessmentCriterion"],
        "industryAlignedActivities": input_json["topicWizardData"]["industryAlignedActivities"],
        "level": input_json["topicWizardData"]["lessonInformation"]["level"],
        "flowProperties_purposes": []
    }

    for flow in input_json["topicWizardData"]["simulationFlow"]:
        if "flowProperties" in flow and "purpose" in flow["flowProperties"]:
            locked_fields["flowProperties_purposes"].append(flow["flowProperties"]["purpose"])
        if "children" in flow:
            for child in flow["children"]:
                if "flowProperties" in child and "purpose" in child["flowProperties"]:
                    locked_fields["flowProperties_purposes"].append(child["flowProperties"]["purpose"])

    transformable_parts = {
        "simulationName": input_json["topicWizardData"]["simulationName"],
        "selectedScenarioOption": input_json["topicWizardData"]["selectedScenarioOption"],
        "workplaceScenario": input_json["topicWizardData"]["workplaceScenario"],
        "simulationFlow": input_json["topicWizardData"]["simulationFlow"]
    }

    # Stream execution (console output commented out for cleaner logs)
    # print("\nStreaming parallel chunk generation:\n" + "-" * 60)
    final_state = None

    for event in app.stream({
        "input_json": input_json,
        "target_scenario": target_scenario,
        "locked_fields": locked_fields,
        "learning_context": {},
        "transformable_parts": transformable_parts,
        "chunks": {},
        "transformed_chunks": {},
        "output_json": {},
        "errors": [],
        "retry_count": 0,
        "start_time": start,
        "validation_report": {}
    }, stream_mode="updates"):
        # Silently process events without console output
        # for node_name, node_output in event.items():
        #     print(f"[OK] {node_name} completed", flush=True)
        final_state = list(event.values())[0] if event else final_state  # Capture last state

    # print("-" * 60)
    runtime_ms = (time.time() - start) * 1000

    # Save output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_state["output_json"], f, indent=2, ensure_ascii=False)

    # Generate comprehensive validation reports
    logger.info("Generating comprehensive validation reports...")
    validation_report = generate_validation_reports(final_state, runtime_ms)

    # Final summary
    print(f"\n{'='*60}")
    print(f"Complete in {runtime_ms:.0f}ms")
    print(f"Status: {'PASS' if not final_state['errors'] else 'FAIL'}")
    print(f"Attempts: {final_state['retry_count'] + 1}")
    if final_state['errors']:
        print(f"Errors: {final_state['errors']}")
    print(f"Output: {output_path}")
    print(f"Validation reports: validation_report.json, validation_report.md")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python main.py <input.json> <scenario_index>")
        sys.exit(1)

    run(sys.argv[1], int(sys.argv[2]))
