"""
Dataset and data loading for Stanford RNA 3D Folding Part 2.

Handles:
- Loading RNA sequences from CSV
- Encoding nucleotide sequences
- Loading 3D coordinate labels from PDB/mmCIF files
- Batching with padding for variable-length sequences
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Nucleotide vocabulary: A, C, G, U + padding + unknown
NUC_VOCAB = {"<pad>": 0, "A": 1, "C": 2, "G": 3, "U": 4, "<unk>": 5}
VOCAB_SIZE = len(NUC_VOCAB)


def encode_sequence(seq: str) -> list[int]:
    """Encode an RNA sequence string to integer tokens."""
    return [NUC_VOCAB.get(c.upper(), NUC_VOCAB["<unk>"]) for c in seq]


def parse_pdb_c1_coords(pdb_path: str) -> np.ndarray:
    """
    Extract C1' atom coordinates from a PDB file.

    Returns:
        np.ndarray of shape (n_residues, 3) with x, y, z coordinates.
    """
    coords = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                if atom_name == "C1'":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)


def parse_mmcif_c1_coords(cif_path: str) -> np.ndarray:
    """
    Extract C1' atom coordinates from an mmCIF file.

    Returns:
        np.ndarray of shape (n_residues, 3) with x, y, z coordinates.
    """
    coords = []
    in_atom_site = False
    col_names = []
    x_idx = y_idx = z_idx = atom_idx = -1

    with open(cif_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("_atom_site."):
                in_atom_site = True
                col_name = line.split(".")[1].split()[0]
                col_names.append(col_name)
                if col_name == "Cartn_x":
                    x_idx = len(col_names) - 1
                elif col_name == "Cartn_y":
                    y_idx = len(col_names) - 1
                elif col_name == "Cartn_z":
                    z_idx = len(col_names) - 1
                elif col_name == "label_atom_id":
                    atom_idx = len(col_names) - 1
            elif in_atom_site and (line.startswith("ATOM") or line.startswith("HETATM")):
                parts = line.split()
                if len(parts) > max(x_idx, y_idx, z_idx, atom_idx):
                    atom_name = parts[atom_idx].strip("'\"")
                    if atom_name == "C1'":
                        x = float(parts[x_idx])
                        y = float(parts[y_idx])
                        z = float(parts[z_idx])
                        coords.append([x, y, z])
            elif in_atom_site and not line.startswith(("ATOM", "HETATM", "_", "#", "loop_")):
                if line and not line.startswith(";"):
                    in_atom_site = False

    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)


class RNATrainDataset(Dataset):
    """
    Training dataset: RNA sequences paired with 3D C1' coordinates.

    Expects a directory structure:
        train_dir/
            structures/       # PDB or mmCIF files
            sequences.csv     # CSV with target_id, sequence columns
    """

    def __init__(self, train_dir: str, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.train_dir = train_dir

        seq_csv = os.path.join(train_dir, "sequences.csv")
        if os.path.exists(seq_csv):
            self.df = pd.read_csv(seq_csv)
        else:
            # Build dataset from structure files
            self.df = self._build_from_structures(train_dir)

        # Filter to sequences within length limit
        self.df = self.df[self.df["sequence"].str.len() <= max_seq_len].reset_index(drop=True)

    def _build_from_structures(self, train_dir: str) -> pd.DataFrame:
        """Build a DataFrame by scanning structure files for sequences."""
        struct_dir = os.path.join(train_dir, "structures")
        records = []
        if os.path.exists(struct_dir):
            for fname in os.listdir(struct_dir):
                if fname.endswith((".pdb", ".cif")):
                    target_id = os.path.splitext(fname)[0]
                    fpath = os.path.join(struct_dir, fname)
                    # Extract sequence from structure
                    seq = self._extract_sequence_from_pdb(fpath) if fname.endswith(".pdb") else ""
                    if seq:
                        records.append({"target_id": target_id, "sequence": seq, "structure_file": fname})
        return pd.DataFrame(records) if records else pd.DataFrame(columns=["target_id", "sequence", "structure_file"])

    @staticmethod
    def _extract_sequence_from_pdb(pdb_path: str) -> str:
        """Extract RNA sequence from PDB by reading C1' atom residue names."""
        res_map = {"A": "A", "C": "C", "G": "G", "U": "U",
                   "DA": "A", "DC": "C", "DG": "G", "DT": "U",
                   "ADE": "A", "CYT": "C", "GUA": "G", "URA": "U"}
        residues = []
        seen = set()
        with open(pdb_path, "r") as f:
            for line in f:
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
        residues.sort()
        return "".join(r[1] for r in residues)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["sequence"]
        target_id = row["target_id"]

        # Encode sequence
        tokens = encode_sequence(seq)
        seq_len = len(tokens)

        # Pad to max_seq_len
        padded = tokens + [0] * (self.max_seq_len - seq_len)
        mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

        # Load coordinates
        coords = np.zeros((self.max_seq_len, 3), dtype=np.float32)
        struct_dir = os.path.join(self.train_dir, "structures")
        for ext in [".pdb", ".cif"]:
            struct_path = os.path.join(struct_dir, target_id + ext)
            if os.path.exists(struct_path):
                if ext == ".pdb":
                    raw_coords = parse_pdb_c1_coords(struct_path)
                else:
                    raw_coords = parse_mmcif_c1_coords(struct_path)
                n = min(len(raw_coords), seq_len, self.max_seq_len)
                coords[:n] = raw_coords[:n]
                break

        return {
            "tokens": torch.tensor(padded, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "seq_len": seq_len,
            "target_id": target_id,
        }


class RNATestDataset(Dataset):
    """
    Test dataset: RNA sequences only (no labels).

    Reads test_sequences.csv with columns: target_id, sequence, etc.
    """

    def __init__(self, csv_path: str, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.df = pd.read_csv(csv_path)
        # Ensure required columns
        assert "target_id" in self.df.columns, "test CSV must have 'target_id' column"
        assert "sequence" in self.df.columns, "test CSV must have 'sequence' column"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["sequence"]
        target_id = row["target_id"]

        tokens = encode_sequence(seq)
        seq_len = len(tokens)

        # For test, we may need to handle sequences longer than max_seq_len
        if seq_len > self.max_seq_len:
            # Truncate (in practice, handle with sliding window or chunking)
            tokens = tokens[: self.max_seq_len]
            seq_len = self.max_seq_len

        padded = tokens + [0] * (self.max_seq_len - seq_len)
        mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

        result = {
            "tokens": torch.tensor(padded, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "seq_len": seq_len,
            "target_id": target_id,
            "sequence": seq,
        }

        # Include extra info if available
        if "description" in self.df.columns:
            result["description"] = row.get("description", "")
        if "all_sequences" in self.df.columns:
            result["all_sequences"] = row.get("all_sequences", "")

        return result


def get_train_dataloader(train_dir: str, max_seq_len: int = 512,
                         batch_size: int = 4, num_workers: int = 4) -> DataLoader:
    dataset = RNATrainDataset(train_dir, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def get_test_dataloader(csv_path: str, max_seq_len: int = 512,
                        batch_size: int = 1, num_workers: int = 0) -> DataLoader:
    dataset = RNATestDataset(csv_path, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)
