"""
Structured lookup layer over the scraped IBM + Broadcom compatibility data.

Why structured lookup AND RAG?
- The compatibility matrices are tabular, exact data (product X is/isn't
  compatible with z/OS version Y). A vector search over free text is the
  WRONG tool for "give me the exact YES/NO list" -- use pandas filtering.
- RAG is used on top for the surrounding NOTES / caveats / narrative
  (PTF requirements, end-of-service info, docs text) where free-text
  retrieval genuinely helps the LLM write a better, more accurate summary.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass

import pandas as pd

ZOS_VERSION_PATTERN = re.compile(r"z/?os\s*v?(\d+)r(\d+)", re.IGNORECASE)


def normalize_zos_version(raw: str) -> str:
    """Normalize things like 'zos 2.5', 'z/OS V2R5', '2.5', 'V2R5' -> 'Z/OS V2R5'."""
    raw = raw.strip()
    m = ZOS_VERSION_PATTERN.search(raw.replace(".", "R").replace("-", "R"))
    if m:
        return f"Z/OS V{m.group(1)}R{m.group(2)}"
    # fall back: "2.5" or "2R5" style
    m2 = re.search(r"(\d)[.\sRr]?(\d)", raw)
    if m2:
        return f"Z/OS V{m2.group(1)}R{m2.group(2)}"
    return raw.upper()


@dataclass
class CompatibilityDB:
    df: pd.DataFrame

    @classmethod
    def load(cls, csv_glob_patterns: list[str]) -> "CompatibilityDB":
        frames = []
        for pattern in csv_glob_patterns:
            for path in glob.glob(pattern):
                frames.append(pd.read_csv(path))
        if not frames:
            raise FileNotFoundError(f"No compatibility CSVs found for patterns: {csv_glob_patterns}")
        df = pd.concat(frames, ignore_index=True)
        df["item_norm"] = df["item"].apply(
            lambda x: normalize_zos_version(x) if isinstance(x, str) and "z/os" in x.lower() else x
        )
        return cls(df=df)

    def products_compatible_with(
        self,
        zos_version: str,
        product_filter: str | None = None,
        vendor_filter: str | None = None,
    ) -> pd.DataFrame:
        """Return all rows where `item` is the given z/OS version and status == YES."""
        target = normalize_zos_version(zos_version)
        mask = (self.df["item_norm"] == target) & (self.df["status"].str.upper() == "YES")
        if vendor_filter:
            mask &= self.df["vendor"].str.lower() == vendor_filter.lower()
        if product_filter:
            mask &= self.df["product_name"].str.contains(product_filter, case=False, na=False)
        cols = ["vendor", "product_name", "product_version", "category", "item", "status", "notes", "source_url"]
        return self.df.loc[mask, cols].drop_duplicates().sort_values(["vendor", "product_name"])

    def product_status_for_version(self, product_name: str, zos_version: str) -> pd.DataFrame:
        """All compatibility rows (any status) for a specific product against a z/OS version."""
        target = normalize_zos_version(zos_version)
        mask = (
            self.df["product_name"].str.contains(product_name, case=False, na=False)
            & (self.df["item_norm"] == target)
        )
        cols = ["vendor", "product_name", "product_version", "category", "item", "status", "notes", "source_url"]
        return self.df.loc[mask, cols].drop_duplicates()

    def known_products(self) -> list[str]:
        return sorted(self.df["product_name"].dropna().unique().tolist())

    def known_zos_versions(self) -> list[str]:
        vals = self.df.loc[self.df["item"].str.contains("z/os", case=False, na=False), "item_norm"]
        return sorted(vals.dropna().unique().tolist())
