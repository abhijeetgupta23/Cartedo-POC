"""
Illustrative Diff Script
Generates a side-by-side comparison of input vs output scenarios
"""

import json
import sys
from deepdiff import DeepDiff
from typing import Dict, Any


def extract_key_fields(json_data: dict) -> Dict[str, Any]:
    """Extract key scenario-dependent fields for comparison"""
    return {
        "simulationName": json_data["topicWizardData"]["simulationName"],
        "selectedScenarioOption": json_data["topicWizardData"]["selectedScenarioOption"][:200],
        "organizationName": json_data["topicWizardData"]["workplaceScenario"]["background"]["organizationName"],
        "aboutOrganization": json_data["topicWizardData"]["workplaceScenario"]["background"]["aboutOrganization"][:200],
        "organizationImageKeyWords": json_data["topicWizardData"]["workplaceScenario"]["background"]["organizationImageKeyWords"],
        "currentIssue": json_data["topicWizardData"]["workplaceScenario"]["challenge"]["currentIssue"][:200],
        "reportingManagerName": json_data["topicWizardData"]["workplaceScenario"]["learnerRoleReportingManager"]["reportingManager"]["name"],
        "reportingManagerEmail": json_data["topicWizardData"]["workplaceScenario"]["learnerRoleReportingManager"]["reportingManager"]["email"],
    }


def generate_illustrative_diff(input_path: str, output_path: str):
    """Generate side-by-side comparison report"""

    # Load JSONs
    with open(input_path, 'r', encoding='utf-8') as f:
        input_json = json.load(f)

    with open(output_path, 'r', encoding='utf-8') as f:
        output_json = json.load(f)

    # Extract key fields
    input_fields = extract_key_fields(input_json)
    output_fields = extract_key_fields(output_json)

    # Generate Markdown report
    md_lines = [
        "# Illustrative Transformation: Before vs After",
        "",
        "This report shows key changes between the original scenario and the transformed scenario.",
        "",
        "---",
        "",
        "## Scenario Overview",
        "",
        "### Original Scenario",
        f"**Organization:** {input_fields['organizationName']}",
        f"**Scenario:** {input_fields['selectedScenarioOption']}...",
        "",
        "### Target Scenario",
        f"**Organization:** {output_fields['organizationName']}",
        f"**Scenario:** {output_fields['selectedScenarioOption']}...",
        "",
        "---",
        "",
        "## Side-by-Side Field Comparison",
        "",
        "| Field | Original | Transformed |",
        "|-------|----------|-------------|",
    ]

    # Add comparisons
    for key in input_fields.keys():
        input_val = str(input_fields[key])[:80].replace("\n", " ")
        output_val = str(output_fields[key])[:80].replace("\n", " ")

        status = "✅ Changed" if input_val != output_val else "⚠️ Unchanged"
        md_lines.append(f"| **{key}** | {input_val}... | {output_val}... |")

    md_lines.extend([
        "",
        "---",
        "",
        "## Detailed Transformations",
        ""
    ])

    # Detailed changes
    details = [
        ("Simulation Name", "simulationName"),
        ("Organization Name", "organizationName"),
        ("Organization Description", "aboutOrganization"),
        ("Organization Keywords", "organizationImageKeyWords"),
        ("Current Business Issue", "currentIssue"),
        ("Reporting Manager Name", "reportingManagerName"),
        ("Reporting Manager Email", "reportingManagerEmail"),
    ]

    for label, key in details:
        md_lines.append(f"### {label}")
        md_lines.append("")
        md_lines.append("**Before:**")
        md_lines.append(f"```\n{input_fields[key]}\n```")
        md_lines.append("")
        md_lines.append("**After:**")
        md_lines.append(f"```\n{output_fields[key]}\n```")
        md_lines.append("")

        if input_fields[key] == output_fields[key]:
            md_lines.append("⚠️ **Warning:** This field was not transformed!")
        else:
            md_lines.append("✅ **Successfully transformed**")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    # Deep diff summary
    diff = DeepDiff(input_json, output_json, ignore_order=False)

    if 'values_changed' in diff:
        num_changes = len(diff['values_changed'])
        md_lines.extend([
            "## Overall Statistics",
            "",
            f"- **Total fields changed:** {num_changes}",
            f"- **Original organization:** {input_fields['organizationName']}",
            f"- **New organization:** {output_fields['organizationName']}",
            f"- **Transformation status:** {'✅ PASS' if num_changes > 0 else '❌ FAIL (no changes detected)'}",
            ""
        ])

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("*This report demonstrates the scenario-aware re-contextualization capabilities of the system.*")

    # Save report
    output_report = "illustrative_diff.md"
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[PASS] Illustrative diff report generated: {output_report}")

    # Print summary to console
    print("\n" + "="*60)
    print("TRANSFORMATION SUMMARY")
    print("="*60)
    print(f"Original:  {input_fields['organizationName']}")
    print(f"Target:    {output_fields['organizationName']}")
    print(f"Changes:   {num_changes} fields transformed")
    print("="*60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python illustrative_diff.py <input.json> <output.json>")
        sys.exit(1)

    generate_illustrative_diff(sys.argv[1], sys.argv[2])
