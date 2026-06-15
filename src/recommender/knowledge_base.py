import os
from pathlib import Path
import pandas as pd

def load_knowledge_base(output_dir: Path | str = "outputs/recommender") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads catalog and mappings from data/recommender/intervention_catalog.csv,
    generates mapping_df dynamically, and writes them to output_dir.
    """
    project_root = Path(__file__).resolve().parents[2]
    catalog_path = project_root / "data" / "recommender" / "intervention_catalog.csv"
    
    if not catalog_path.exists():
        # Create default folder and file if it does not exist
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        # Write default lines
        default_content = (
            "item_id,intervention_name,description,target_risks,difficulty_level,estimated_hours_per_week,recommended_phase,expected_effect,prerequisite_level\n"
            "attendance_monitoring,Daily Attendance Monitoring,Sign-in sheets and weekly check-ins.,R3_ATTENDANCE_RISK,1,0.5,Stabilize,0.9,0\n"
        )
        catalog_path.write_text(default_content, encoding="utf-8")
        
    catalog_df = pd.read_csv(catalog_path)
    
    # Generate mapping_df dynamically from catalog_df target_risks
    mappings = []
    for _, row in catalog_df.iterrows():
        item_id = row["item_id"]
        target_risks_str = str(row.get("target_risks", ""))
        if pd.isna(row.get("target_risks")):
            target_risks_str = ""
        target_risks = [r.strip() for r in target_risks_str.split(",") if r.strip()]
        for risk in target_risks:
            mappings.append({"risk_code": risk, "item_id": item_id})
            
    mapping_df = pd.DataFrame(mappings)
    
    # Save a copy to output_dir as required by pipeline/tests
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    catalog_df.to_csv(out_path / "intervention_catalog.csv", index=False)
    mapping_df.to_csv(out_path / "risk_intervention_mapping.csv", index=False)
    
    return catalog_df, mapping_df

def initialize_knowledge_base(output_dir: Path | str = "outputs/recommender") -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_knowledge_base(output_dir)
