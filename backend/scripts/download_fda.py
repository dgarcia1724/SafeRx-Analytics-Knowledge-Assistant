"""
Download FDA drug labels from the openFDA API.

Downloads 25 commonly prescribed drugs with clean section parsing.

Usage:
    python scripts/download_fda.py
"""

import httpx
import json
import time
import re
from pathlib import Path
from typing import Optional


# Target drugs - 25 commonly prescribed medications organized by category
TARGET_DRUGS = {
    # Pain/Inflammation
    "aspirin": "pain_inflammation",
    "ibuprofen": "pain_inflammation",
    "acetaminophen": "pain_inflammation",
    "naproxen": "pain_inflammation",

    # Cardiovascular
    "lisinopril": "cardiovascular",
    "metoprolol": "cardiovascular",
    "amlodipine": "cardiovascular",
    "warfarin": "cardiovascular",
    "atorvastatin": "cardiovascular",
    "losartan": "cardiovascular",

    # Diabetes
    "metformin": "diabetes",
    "glipizide": "diabetes",

    # Mental Health
    "sertraline": "mental_health",
    "fluoxetine": "mental_health",
    "alprazolam": "mental_health",
    "gabapentin": "mental_health",

    # Antibiotics
    "amoxicillin": "antibiotics",
    "azithromycin": "antibiotics",
    "ciprofloxacin": "antibiotics",

    # Common/Other
    "omeprazole": "gastrointestinal",
    "levothyroxine": "thyroid",
    "prednisone": "corticosteroid",
    "albuterol": "respiratory",
    "hydrochlorothiazide": "diuretic",
    "furosemide": "diuretic",
}

# openFDA API endpoint
FDA_API_URL = "https://api.fda.gov/drug/label.json"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "documents" / "drug_labels"

# Structured output directory for parsed JSON
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed_labels"


# Section keys with clean headers
SECTION_MAPPING = {
    "indications_and_usage": "INDICATIONS AND USAGE",
    "dosage_and_administration": "DOSAGE AND ADMINISTRATION",
    "dosage_forms_and_strengths": "DOSAGE FORMS AND STRENGTHS",
    "contraindications": "CONTRAINDICATIONS",
    "warnings_and_precautions": "WARNINGS AND PRECAUTIONS",
    "warnings": "WARNINGS",
    "precautions": "PRECAUTIONS",
    "adverse_reactions": "ADVERSE REACTIONS",
    "drug_interactions": "DRUG INTERACTIONS",
    "use_in_specific_populations": "USE IN SPECIFIC POPULATIONS",
    "pregnancy": "PREGNANCY",
    "nursing_mothers": "NURSING MOTHERS",
    "pediatric_use": "PEDIATRIC USE",
    "geriatric_use": "GERIATRIC USE",
    "overdosage": "OVERDOSAGE",
    "description": "DESCRIPTION",
    "clinical_pharmacology": "CLINICAL PHARMACOLOGY",
    "mechanism_of_action": "MECHANISM OF ACTION",
    "pharmacodynamics": "PHARMACODYNAMICS",
    "pharmacokinetics": "PHARMACOKINETICS",
    "boxed_warning": "BOXED WARNING",
    "how_supplied": "HOW SUPPLIED",
    "storage_and_handling": "STORAGE AND HANDLING",
    "patient_counseling_information": "PATIENT COUNSELING INFORMATION",
}

# Priority sections for RAG (these are most commonly queried)
PRIORITY_SECTIONS = [
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings_and_precautions",
    "adverse_reactions",
    "drug_interactions",
    "use_in_specific_populations",
    "boxed_warning",
]


def clean_text(text: str) -> str:
    """Clean FDA label text by removing excessive whitespace and formatting."""
    if not text:
        return ""

    # Remove bullet point markers that are inconsistent
    text = re.sub(r'^\s*[\u2022\u2023\u25E6\u2043\u2219]\s*', '- ', text, flags=re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Clean up lines
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    text = '\n'.join(cleaned_lines)

    return text.strip()


def extract_section_content(label_data: dict, section_key: str) -> Optional[str]:
    """Extract and clean content from a specific section."""
    content = label_data.get(section_key)

    if not content:
        return None

    if isinstance(content, list):
        content = "\n\n".join(content)

    cleaned = clean_text(content)
    return cleaned if cleaned else None


def fetch_drug_label(drug_name: str) -> Optional[dict]:
    """Fetch drug label from openFDA API."""
    # Try generic name first
    params = {
        "search": f'openfda.generic_name:"{drug_name}"',
        "limit": 1,
    }

    try:
        response = httpx.get(FDA_API_URL, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            return data["results"][0]

        # Try brand name if generic fails
        params["search"] = f'openfda.brand_name:"{drug_name}"'
        response = httpx.get(FDA_API_URL, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            return data["results"][0]

        # Try substance name as last resort
        params["search"] = f'openfda.substance_name:"{drug_name}"'
        response = httpx.get(FDA_API_URL, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            return data["results"][0]

        return None

    except httpx.HTTPStatusError as e:
        print(f"HTTP error for {drug_name}: {e}")
        return None
    except Exception as e:
        print(f"Error fetching {drug_name}: {e}")
        return None


def get_drug_info(label_data: dict) -> dict:
    """Extract drug identification information."""
    openfda = label_data.get("openfda", {})

    return {
        "generic_name": openfda.get("generic_name", ["Unknown"])[0],
        "brand_names": openfda.get("brand_name", []),
        "manufacturer": openfda.get("manufacturer_name", ["Unknown"])[0] if openfda.get("manufacturer_name") else "Unknown",
        "product_type": openfda.get("product_type", ["Unknown"])[0] if openfda.get("product_type") else "Unknown",
        "route": openfda.get("route", ["Unknown"])[0] if openfda.get("route") else "Unknown",
        "substance_name": openfda.get("substance_name", ["Unknown"])[0] if openfda.get("substance_name") else "Unknown",
        "pharm_class_epc": openfda.get("pharm_class_epc", []),
        "pharm_class_moa": openfda.get("pharm_class_moa", []),
    }


def parse_label_to_structured(label_data: dict, drug_name: str, category: str) -> dict:
    """Parse FDA label into structured JSON format."""
    drug_info = get_drug_info(label_data)

    sections = {}
    for key, header in SECTION_MAPPING.items():
        content = extract_section_content(label_data, key)
        if content:
            sections[key] = {
                "header": header,
                "content": content,
                "priority": key in PRIORITY_SECTIONS,
            }

    return {
        "drug_name": drug_name,
        "category": category,
        "drug_info": drug_info,
        "sections": sections,
        "source": "FDA Drug Label via openFDA API",
        "api_version": label_data.get("version", "unknown"),
    }


def extract_label_text(label_data: dict, drug_name: str, category: str) -> str:
    """Extract readable text from FDA label JSON."""
    drug_info = get_drug_info(label_data)

    lines = [
        f"DRUG: {drug_info['generic_name'].upper()}",
        f"Category: {category.replace('_', ' ').title()}",
        f"Brand Names: {', '.join(drug_info['brand_names']) if drug_info['brand_names'] else 'N/A'}",
        f"Route: {drug_info['route']}",
        f"Source: FDA Drug Label",
        "",
        "=" * 60,
        "",
    ]

    # Add sections in priority order first, then others
    added_sections = set()

    # Priority sections first
    for section_key in PRIORITY_SECTIONS:
        content = extract_section_content(label_data, section_key)
        if content:
            header = SECTION_MAPPING.get(section_key, section_key.upper())
            lines.extend([
                header,
                "-" * len(header),
                "",
                content,
                "",
                "",
            ])
            added_sections.add(section_key)

    # Then other sections
    for section_key, header in SECTION_MAPPING.items():
        if section_key not in added_sections:
            content = extract_section_content(label_data, section_key)
            if content:
                lines.extend([
                    header,
                    "-" * len(header),
                    "",
                    content,
                    "",
                    "",
                ])

    return "\n".join(lines)


def main():
    """Download FDA drug labels for target drugs."""
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading FDA drug labels")
    print(f"  Text output: {OUTPUT_DIR}")
    print(f"  Parsed JSON: {PARSED_DIR}")
    print(f"  Target drugs: {len(TARGET_DRUGS)}")
    print("-" * 60)

    successful = 0
    failed = []
    all_parsed = {}

    for drug_name, category in TARGET_DRUGS.items():
        print(f"[{category:15}] {drug_name:20}", end=" ")

        label_data = fetch_drug_label(drug_name)

        if label_data:
            # Parse to structured JSON
            parsed = parse_label_to_structured(label_data, drug_name, category)
            all_parsed[drug_name] = parsed

            # Extract readable text
            label_text = extract_label_text(label_data, drug_name, category)

            # Save text file
            filename = f"{drug_name.lower().replace(' ', '_')}_label.txt"
            filepath = OUTPUT_DIR / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(label_text)

            # Save individual parsed JSON
            json_filename = f"{drug_name.lower().replace(' ', '_')}_parsed.json"
            json_filepath = PARSED_DIR / json_filename

            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)

            # Save raw JSON for reference
            raw_json_filename = f"{drug_name.lower().replace(' ', '_')}_raw.json"
            raw_json_filepath = PARSED_DIR / raw_json_filename

            with open(raw_json_filepath, "w", encoding="utf-8") as f:
                json.dump(label_data, f, indent=2)

            section_count = len(parsed["sections"])
            print(f"OK ({section_count} sections)")
            successful += 1
        else:
            print("FAILED")
            failed.append(drug_name)

        # Rate limiting - openFDA allows 240 requests/minute without API key
        time.sleep(0.3)

    # Save combined parsed data
    combined_filepath = PARSED_DIR / "all_drugs_parsed.json"
    with open(combined_filepath, "w", encoding="utf-8") as f:
        json.dump(all_parsed, f, indent=2)

    # Summary
    print("-" * 60)
    print(f"Successfully downloaded: {successful}/{len(TARGET_DRUGS)}")

    if failed:
        print(f"Failed drugs: {', '.join(failed)}")
        print("\nNote: Some drugs may not be available in openFDA or may have")
        print("different naming conventions. Manual review recommended.")

    # Print category breakdown
    print("\nCategory breakdown:")
    categories = {}
    for drug, cat in TARGET_DRUGS.items():
        if drug not in failed:
            categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        print(f"  {cat:20}: {count} drugs")


if __name__ == "__main__":
    main()
