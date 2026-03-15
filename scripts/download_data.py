"""
Download and prepare training data for RNA 3D Folding.

Sources:
- PDB/RNA structures from RCSB PDB
- Competition data from Kaggle
- RNA sequence databases

Usage:
    python scripts/download_data.py --source pdb --output data/train
    python scripts/download_data.py --source kaggle --output data/
"""

import os
import argparse
import json
import subprocess
from pathlib import Path


def download_pdb_rna_structures(output_dir: str, max_structures: int = 1000):
    """
    Download RNA structures from RCSB PDB using their REST API.

    Fetches structures containing RNA chains with resolution < 4.0 Angstroms.
    """
    struct_dir = os.path.join(output_dir, "structures")
    os.makedirs(struct_dir, exist_ok=True)

    # RCSB PDB search query for RNA structures
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "RNA"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 4.0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_structures},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}]
        }
    }

    print(f"Searching RCSB PDB for RNA structures (max {max_structures})...")
    print("Query JSON saved. Use the following curl command to fetch PDB IDs:")
    print()

    query_file = os.path.join(output_dir, "pdb_query.json")
    with open(query_file, "w") as f:
        json.dump(query, f, indent=2)

    print(f"  curl -X POST -H 'Content-Type: application/json' \\")
    print(f"    -d @{query_file} \\")
    print(f"    'https://search.rcsb.org/rcsbsearch/v2/query'")
    print()
    print("Then download individual structures with:")
    print("  curl -o XXXX.cif 'https://files.rcsb.org/download/XXXX.cif'")
    print()

    # Create a helper download script
    dl_script = os.path.join(output_dir, "download_pdbs.sh")
    with open(dl_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Download RNA structures from RCSB PDB\n")
        f.write(f"# Output directory: {struct_dir}\n\n")
        f.write("RESULT=$(curl -s -X POST -H 'Content-Type: application/json' \\\n")
        f.write(f"  -d @{query_file} \\\n")
        f.write("  'https://search.rcsb.org/rcsbsearch/v2/query')\n\n")
        f.write("PDBIDS=$(echo $RESULT | python3 -c \"import sys,json; d=json.load(sys.stdin); print(' '.join(r['identifier'] for r in d.get('result_set',[])))\")\n\n")
        f.write("for PDB_ID in $PDBIDS; do\n")
        f.write(f"  echo \"Downloading $PDB_ID...\"\n")
        f.write(f"  curl -s -o {struct_dir}/$PDB_ID.cif \"https://files.rcsb.org/download/$PDB_ID.cif\"\n")
        f.write("done\n")
        f.write(f"\necho \"Done. Files saved to {struct_dir}/\"\n")

    os.chmod(dl_script, 0o755)
    print(f"Download script written to: {dl_script}")


def download_kaggle_data(output_dir: str):
    """Download competition data using Kaggle CLI."""
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading competition data from Kaggle...")
    print("Make sure you have:")
    print("  1. kaggle CLI installed: pip install kaggle")
    print("  2. API credentials at ~/.kaggle/kaggle.json")
    print("  3. Accepted competition rules on kaggle.com")
    print()

    cmd = [
        "kaggle", "competitions", "download",
        "-c", "stanford-rna-3d-folding-2",
        "-p", output_dir
    ]
    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"Data downloaded to {output_dir}")

        # Unzip if needed
        for f in os.listdir(output_dir):
            if f.endswith(".zip"):
                print(f"Unzipping {f}...")
                subprocess.run(["unzip", "-o", os.path.join(output_dir, f), "-d", output_dir], check=True)
    except FileNotFoundError:
        print("ERROR: kaggle CLI not found. Install with: pip install kaggle")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Download failed: {e}")


def create_sequences_csv(train_dir: str):
    """
    Create sequences.csv from downloaded structure files.

    Scans PDB/mmCIF files and extracts RNA sequences.
    """
    struct_dir = os.path.join(train_dir, "structures")
    if not os.path.exists(struct_dir):
        print(f"No structures directory found at {struct_dir}")
        return

    records = []
    for fname in sorted(os.listdir(struct_dir)):
        if not fname.endswith((".pdb", ".cif")):
            continue
        target_id = os.path.splitext(fname)[0]
        fpath = os.path.join(struct_dir, fname)

        # Extract RNA sequence from structure file
        seq = extract_rna_sequence(fpath)
        if seq and len(seq) >= 10:
            records.append({"target_id": target_id, "sequence": seq, "structure_file": fname})

    if records:
        import pandas as pd
        df = pd.DataFrame(records)
        csv_path = os.path.join(train_dir, "sequences.csv")
        df.to_csv(csv_path, index=False)
        print(f"Created {csv_path} with {len(df)} entries")
        print(f"  Sequence lengths: min={df['sequence'].str.len().min()}, "
              f"max={df['sequence'].str.len().max()}, "
              f"mean={df['sequence'].str.len().mean():.0f}")
    else:
        print("No valid RNA sequences found in structure files.")


def extract_rna_sequence(filepath: str) -> str:
    """Extract RNA sequence from PDB or mmCIF file."""
    res_map = {"A": "A", "C": "C", "G": "G", "U": "U",
               "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U"}
    residues = []
    seen = set()

    with open(filepath, "r") as f:
        for line in f:
            if filepath.endswith(".pdb"):
                if line.startswith(("ATOM", "HETATM")):
                    atom_name = line[12:16].strip()
                    if atom_name == "C1'":
                        resname = line[17:20].strip()
                        resseq = line[22:26].strip()
                        chain = line[21]
                        key = (chain, resseq)
                        if key not in seen:
                            seen.add(key)
                            nuc = res_map.get(resname, "")
                            if nuc:
                                residues.append((int(resseq), nuc))
            # mmCIF parsing is more complex; simplified version
            elif filepath.endswith(".cif"):
                if line.startswith("ATOM") and "C1'" in line:
                    parts = line.split()
                    if len(parts) > 5:
                        resname = parts[5] if len(parts) > 5 else ""
                        nuc = res_map.get(resname, "")
                        if nuc:
                            resseq = parts[8] if len(parts) > 8 else str(len(residues))
                            key = (parts[6] if len(parts) > 6 else "", resseq)
                            if key not in seen:
                                seen.add(key)
                                try:
                                    residues.append((int(resseq), nuc))
                                except ValueError:
                                    residues.append((len(residues), nuc))

    residues.sort()
    return "".join(r[1] for r in residues)


def main():
    parser = argparse.ArgumentParser(description="Download RNA 3D Folding data")
    parser.add_argument("--source", choices=["pdb", "kaggle", "both"], default="both")
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--max-structures", type=int, default=1000)
    args = parser.parse_args()

    if args.source in ("pdb", "both"):
        train_dir = os.path.join(args.output, "train")
        download_pdb_rna_structures(train_dir, args.max_structures)
        print()

    if args.source in ("kaggle", "both"):
        download_kaggle_data(args.output)
        print()

    # Create sequences CSV if structure files exist
    train_dir = os.path.join(args.output, "train")
    if os.path.exists(os.path.join(train_dir, "structures")):
        create_sequences_csv(train_dir)


if __name__ == "__main__":
    main()
