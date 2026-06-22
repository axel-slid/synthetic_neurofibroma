#!/usr/bin/env python3
"""Build a curated neurofibroma literature package.

The script fetches PubMed metadata for a hand-curated set of papers and merges
it with modeling notes that are specific to NF1-associated neurofibroma growth.
Outputs are written under literature/data and literature/visualizations.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import textwrap
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from plotly.utils import PlotlyJSONEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
VIZ_DIR = ROOT / "visualizations"


CURATED = [
    {
        "pmid": "25877329",
        "category": "genetics_signaling",
        "evidence_type": "review",
        "key_findings": "NF1 is a tumor suppressor and RASopathy gene; neurofibromin loss dysregulates RAS/MAPK and other growth-control programs.",
        "physical_model_use": "Defines the intracellular growth-state switch for NF1-deficient Schwann-lineage agents.",
        "spatial_scale": 0.3,
        "model_readiness": 3.0,
    },
    {
        "pmid": "26860753",
        "category": "genetics_signaling",
        "evidence_type": "review",
        "key_findings": "Summarizes neurofibromin as a multifunctional tumor suppressor with RAS-GAP and non-RAS signaling roles.",
        "physical_model_use": "Supports representing NF1 loss as a signaling-state change rather than only a proliferation constant.",
        "spatial_scale": 0.3,
        "model_readiness": 2.8,
    },
    {
        "pmid": "11988578",
        "category": "cell_origin",
        "evidence_type": "mouse genetics",
        "key_findings": "Classic evidence that neurofibroma initiation involves Schwann-cell NF1 loss and a permissive tumor environment.",
        "physical_model_use": "Makes Schwann-lineage cells the initiating tumor cell class in a physical model.",
        "spatial_scale": 1.0,
        "model_readiness": 4.0,
    },
    {
        "pmid": "18984156",
        "category": "microenvironment",
        "evidence_type": "mouse genetics",
        "key_findings": "Nf1-dependent tumors required an Nf1+/- and c-kit-dependent bone marrow/microenvironment component.",
        "physical_model_use": "Adds host stromal/hematopoietic recruitment as a required state variable, not a passive background.",
        "spatial_scale": 2.3,
        "model_readiness": 4.0,
    },
    {
        "pmid": "19427294",
        "category": "cell_origin",
        "evidence_type": "mouse genetics",
        "key_findings": "Dermal neurofibroma formation was linked to susceptible skin-derived precursor/Schwann-lineage populations and local microenvironment.",
        "physical_model_use": "Separates cutaneous/dermal origin assumptions from plexiform origin assumptions.",
        "spatial_scale": 1.5,
        "model_readiness": 4.0,
    },
    {
        "pmid": "21551250",
        "category": "cell_origin",
        "evidence_type": "mouse genetics",
        "key_findings": "Defined developmental windows in Schwann cells that are susceptible for plexiform neurofibroma development.",
        "physical_model_use": "Motivates an age/development-dependent initiation probability for plexiform tumors.",
        "spatial_scale": 1.5,
        "model_readiness": 4.2,
    },
    {
        "pmid": "25446898",
        "category": "cell_origin",
        "evidence_type": "mouse genetics",
        "key_findings": "Identified embryonic nerve-root Schwann-lineage cells as cells of origin for NF1-associated plexiform neurofibroma.",
        "physical_model_use": "Places plexiform tumor seeds on embryonic nerve-root/peripheral nerve topology.",
        "spatial_scale": 1.7,
        "model_readiness": 4.4,
    },
    {
        "pmid": "30348677",
        "category": "cell_origin",
        "evidence_type": "mouse genetics and transcriptomics",
        "key_findings": "Spatiotemporal NF1 loss in Schwann-lineage cells generated different cutaneous neurofibroma types and implicated Hippo/YAP modification.",
        "physical_model_use": "Links tumor subtype to timing, location, and mechanotransduction-sensitive signaling.",
        "spatial_scale": 1.8,
        "model_readiness": 4.0,
    },
    {
        "pmid": "32642729",
        "category": "cell_origin",
        "evidence_type": "review",
        "key_findings": "Reviews evolving evidence for neurofibroma tumor cells of origin across dermal/cutaneous and plexiform subtypes.",
        "physical_model_use": "Useful guardrail for choosing the correct initiating-cell assumptions by subtype.",
        "spatial_scale": 1.4,
        "model_readiness": 3.5,
    },
    {
        "pmid": "36139671",
        "category": "cell_origin",
        "evidence_type": "review",
        "key_findings": "Synthesizes cellular-origin evidence through Schwann-cell lineage development for NF1 neurofibromas.",
        "physical_model_use": "Maps lineage state to initiation and phenotype in a multi-state growth model.",
        "spatial_scale": 1.4,
        "model_readiness": 3.5,
    },
    {
        "pmid": "17215493",
        "category": "growth_natural_history",
        "evidence_type": "volumetric MRI cohort",
        "key_findings": "Longitudinal volumetric MRI showed heterogeneous plexiform neurofibroma growth and faster growth in younger patients.",
        "physical_model_use": "Provides patient-scale growth-rate calibration targets and a >=20% volume-change convention.",
        "spatial_scale": 4.0,
        "model_readiness": 5.0,
    },
    {
        "pmid": "23035791",
        "category": "growth_natural_history",
        "evidence_type": "whole-body MRI cohort",
        "key_findings": "A 201-patient cohort found internal plexiform neurofibroma burden and growth rates that correlated with younger age and tumor volume.",
        "physical_model_use": "Supports age-dependent and tumor-burden-dependent growth-rate priors.",
        "spatial_scale": 4.2,
        "model_readiness": 5.0,
    },
    {
        "pmid": "36332985",
        "category": "growth_natural_history",
        "evidence_type": "10-year adult whole-body MRI cohort",
        "key_findings": "A decade-long adult cohort showed that internal neurofibroma growth is heterogeneous and often slower in adults than in children.",
        "physical_model_use": "Constrains adult growth-rate priors and supports tumor-specific rather than uniform growth parameters.",
        "spatial_scale": 4.3,
        "model_readiness": 5.0,
    },
    {
        "pmid": "18559970",
        "category": "growth_natural_history",
        "evidence_type": "whole-body MRI cohort",
        "key_findings": "Whole-body MRI was used to quantify benign internal tumor burden in NF1.",
        "physical_model_use": "Defines imaging-derived tumor-burden observables for calibration and validation.",
        "spatial_scale": 4.2,
        "model_readiness": 4.8,
    },
    {
        "pmid": "29718344",
        "category": "growth_natural_history",
        "evidence_type": "natural history cohort",
        "key_findings": "PN volume and changes were evaluated against clinically meaningful morbidities.",
        "physical_model_use": "Connects simulated volume changes to functional outcomes and clinical relevance.",
        "spatial_scale": 4.1,
        "model_readiness": 4.5,
    },
    {
        "pmid": "32152628",
        "category": "growth_natural_history",
        "evidence_type": "longitudinal imaging cohort",
        "key_findings": "Analyzed growth of plexiform neurofibromas and distinct nodular lesions in NF1.",
        "physical_model_use": "Suggests separating diffuse plexiform mass behavior from nodular lesion behavior.",
        "spatial_scale": 4.1,
        "model_readiness": 4.8,
    },
    {
        "pmid": "24321536",
        "category": "growth_natural_history",
        "evidence_type": "retrospective cohort",
        "key_findings": "Tested whether puberty accelerates plexiform neurofibroma growth in NF1.",
        "physical_model_use": "Useful for deciding whether pubertal state should be an explicit growth covariate.",
        "spatial_scale": 4.0,
        "model_readiness": 4.0,
    },
    {
        "pmid": "39497113",
        "category": "growth_natural_history",
        "evidence_type": "pediatric whole-body MRI cohort",
        "key_findings": "Long-term pediatric WBMRI followed patients without initial tumor burden and documented newly developed peripheral nerve sheath tumors.",
        "physical_model_use": "Provides data for de novo tumor appearance probabilities in children.",
        "spatial_scale": 4.2,
        "model_readiness": 4.4,
    },
    {
        "pmid": "20233971",
        "category": "microenvironment",
        "evidence_type": "review",
        "key_findings": "Reviews mast cells, Schwann cells, fibroblasts, blood vessels, and matrix as key neurofibroma microenvironment components.",
        "physical_model_use": "Defines non-tumor cell classes for a multicellular model.",
        "spatial_scale": 2.5,
        "model_readiness": 3.8,
    },
    {
        "pmid": "16835260",
        "category": "microenvironment",
        "evidence_type": "mouse and cellular model",
        "key_findings": "Nf1+/- mast cells induced neurofibroma-like phenotypes through secreted TGF-beta signaling.",
        "physical_model_use": "Adds mast-cell/fibroblast signaling and collagen deposition pathways to the model.",
        "spatial_scale": 2.7,
        "model_readiness": 4.1,
    },
    {
        "pmid": "23099891",
        "category": "microenvironment",
        "evidence_type": "mouse and pharmacology",
        "key_findings": "Neurofibroma-associated macrophages contributed to tumor growth and drug response.",
        "physical_model_use": "Supports macrophage recruitment, density, and state as growth modifiers.",
        "spatial_scale": 2.8,
        "model_readiness": 4.2,
    },
    {
        "pmid": "29596064",
        "category": "microenvironment",
        "evidence_type": "mouse genetics and intervention",
        "key_findings": "Inflammation and tumor microenvironment were functionally linked to neurofibroma tumorigenesis.",
        "physical_model_use": "Supports inflammatory recruitment and cytokine fields after Schwann-cell NF1 loss.",
        "spatial_scale": 2.9,
        "model_readiness": 4.2,
    },
    {
        "pmid": "35589737",
        "category": "microenvironment",
        "evidence_type": "mouse and iPSC models",
        "key_findings": "NF1-mutant neuronal hyperexcitability promoted nervous system tumor progression; peripheral neuron activity-regulated COL1A2 increased NF1-deficient Schwann-cell proliferation.",
        "physical_model_use": "Adds nerve activity and neuron-derived paracrine factors as spatially local growth inputs.",
        "spatial_scale": 2.8,
        "model_readiness": 4.0,
    },
    {
        "pmid": "33413690",
        "category": "mechanics_ecm",
        "evidence_type": "single-cell RNA sequencing",
        "key_findings": "Single-cell analysis mapped the human cutaneous neurofibroma matrisome and matrix-producing cell populations.",
        "physical_model_use": "Defines ECM species and likely cell sources for matrix deposition fields.",
        "spatial_scale": 3.0,
        "model_readiness": 4.0,
    },
    {
        "pmid": "37140985",
        "category": "mechanics_ecm",
        "evidence_type": "ECM biology and treatment response",
        "key_findings": "Basement membrane ECM proteins characterized NF1 neurofibroma development and response to MEK inhibition.",
        "physical_model_use": "Links ECM composition to tumor state and treatment response variables.",
        "spatial_scale": 3.0,
        "model_readiness": 4.0,
    },
    {
        "pmid": "27617404",
        "category": "model_systems",
        "evidence_type": "cell line resource",
        "key_findings": "Established immortalized human normal and NF1 neurofibroma Schwann cell lines.",
        "physical_model_use": "Provides cell resources for parameterizing growth, migration, and drug response.",
        "spatial_scale": 1.2,
        "model_readiness": 4.5,
    },
    {
        "pmid": "29055717",
        "category": "model_systems",
        "evidence_type": "3D culture and drug screening",
        "key_findings": "Developed 3D plexiform neurofibroma culture models for phenotype characterization and drug screening.",
        "physical_model_use": "Directly informs 3D spheroid/co-culture model geometry, matrix proteolysis, and assay outputs.",
        "spatial_scale": 2.5,
        "model_readiness": 5.0,
    },
    {
        "pmid": "29893754",
        "category": "model_systems",
        "evidence_type": "pharmacogenomic resource",
        "key_findings": "Generated pharmacological and genomic profiling data for plexiform neurofibroma-derived Schwann cells.",
        "physical_model_use": "Useful for model priors on pathway dependencies and drug-response parameters.",
        "spatial_scale": 1.2,
        "model_readiness": 4.5,
    },
    {
        "pmid": "30713041",
        "category": "model_systems",
        "evidence_type": "iPSC reprogramming",
        "key_findings": "Reprogramming captured genetic and tumorigenic properties of NF1 plexiform neurofibromas.",
        "physical_model_use": "Provides developmental-cell-state models for initiation and differentiation assumptions.",
        "spatial_scale": 1.7,
        "model_readiness": 4.0,
    },
    {
        "pmid": "33108355",
        "category": "model_systems",
        "evidence_type": "humanized iPSC model",
        "key_findings": "Humanized iPSC-derived neurofibroma models delineated pathogenesis and developmental origins.",
        "physical_model_use": "Useful bridge between developmental origin and xenograft/3D physical systems.",
        "spatial_scale": 2.0,
        "model_readiness": 4.3,
    },
    {
        "pmid": "35172160",
        "category": "model_systems",
        "evidence_type": "iPSC-derived xenograft model",
        "key_findings": "iPSC-derived human neurofibroma-like tumors in mice revealed Schwann-cell heterogeneity within plexiform neurofibromas.",
        "physical_model_use": "Supports multiple Schwann-lineage states in agent-based or hybrid models.",
        "spatial_scale": 2.1,
        "model_readiness": 4.2,
    },
    {
        "pmid": "38744290",
        "category": "model_systems",
        "evidence_type": "patient-derived organoids",
        "key_findings": "Established rapid patient-derived cutaneous neurofibroma organoids for screening.",
        "physical_model_use": "Provides organoid-scale growth/readout systems for cutaneous neurofibroma model validation.",
        "spatial_scale": 2.3,
        "model_readiness": 4.8,
    },
    {
        "pmid": "37330719",
        "category": "model_systems",
        "evidence_type": "review",
        "key_findings": "Reviews existing and emerging preclinical models for NF1-related cutaneous neurofibromas.",
        "physical_model_use": "Helps match experimental validation system to simulated neurofibroma subtype.",
        "spatial_scale": 2.0,
        "model_readiness": 3.8,
    },
    {
        "pmid": "39061138",
        "category": "mechanics_ecm",
        "evidence_type": "3D spheroid and secretome model",
        "key_findings": "Fibroblast-derived secretome stimulated growth and invasiveness of 3D plexiform neurofibroma spheroids.",
        "physical_model_use": "Adds fibroblast paracrine signaling and extracellular vesicle effects to 3D growth rules.",
        "spatial_scale": 2.7,
        "model_readiness": 4.8,
    },
    {
        "pmid": "",
        "doi": "10.3390/cells15100877",
        "manual_title": "Mechanical Stiffening Promotes Growth, Invasion-Associated Phenotypes, and Reduced Selumetinib Sensitivity in 3D Plexiform Neurofibroma Cultures",
        "manual_authors": "Ji K; Shi C; Zhang J; Mattingly RR",
        "manual_journal": "Cells",
        "manual_year": "2026",
        "manual_url": "https://www.mdpi.com/2073-4409/15/10/877",
        "category": "mechanics_ecm",
        "evidence_type": "3D culture mechanobiology",
        "key_findings": "ECM stiffening promoted growth, invasion-associated phenotypes, and reduced selumetinib sensitivity in 3D plexiform neurofibroma cultures.",
        "physical_model_use": "Directly supports ECM stiffness as a mechanical state variable coupled to growth, invasion, and therapy response.",
        "spatial_scale": 3.2,
        "model_readiness": 4.8,
    },
    {
        "pmid": "24166582",
        "category": "malignant_transformation",
        "evidence_type": "clinical imaging cohort",
        "key_findings": "Benign whole-body tumor volume was associated with MPNST risk in NF1.",
        "physical_model_use": "Adds total tumor burden as a risk covariate for malignant-transformation modules.",
        "spatial_scale": 4.3,
        "model_readiness": 4.0,
    },
    {
        "pmid": "21987445",
        "category": "malignant_transformation",
        "evidence_type": "genomic pathology",
        "key_findings": "Atypical neurofibromas were described as premalignant tumors, with CDKN2A/B deletion as an early progression step.",
        "physical_model_use": "Defines a premalignant state transition between benign neurofibroma and MPNST.",
        "spatial_scale": 3.8,
        "model_readiness": 4.0,
    },
    {
        "pmid": "29409029",
        "category": "malignant_transformation",
        "evidence_type": "pathology cohort",
        "key_findings": "Characterized 76 atypical neurofibromas as precursors to NF1-associated MPNST.",
        "physical_model_use": "Supports a distinct nodular/atypical compartment rather than direct benign-to-malignant conversion.",
        "spatial_scale": 3.9,
        "model_readiness": 4.0,
    },
    {
        "pmid": "28592921",
        "category": "malignant_transformation",
        "evidence_type": "review",
        "key_findings": "Reviews clinical and biological insights into MPNST therapy and progression.",
        "physical_model_use": "Frames which model outputs are relevant if simulating malignant transformation.",
        "spatial_scale": 4.0,
        "model_readiness": 3.5,
    },
]


CATEGORY_LABELS = {
    "genetics_signaling": "Genetics/signaling",
    "cell_origin": "Cell of origin",
    "growth_natural_history": "Growth/natural history",
    "microenvironment": "Microenvironment",
    "mechanics_ecm": "Mechanics/ECM",
    "model_systems": "Model systems",
    "malignant_transformation": "Malignant transformation",
}

CATEGORY_COLORS = {
    "genetics_signaling": "#525252",
    "cell_origin": "#2563eb",
    "growth_natural_history": "#16a34a",
    "microenvironment": "#dc2626",
    "mechanics_ecm": "#7c3aed",
    "model_systems": "#0891b2",
    "malignant_transformation": "#b45309",
}

SEARCH_QUERIES = [
    "neurofibroma cell of origin Schwann cell lineage NF1",
    "plexiform neurofibroma growth dynamics volumetric MRI NF1",
    "neurofibroma microenvironment mast cells macrophages fibroblasts extracellular matrix NF1",
    "neurofibroma model physical computational mathematical extracellular matrix stiffness 3D culture",
    "patient-derived cutaneous neurofibroma organoid plexiform neurofibroma 3D culture",
    "NF1 plexiform neurofibroma malignant transformation atypical neurofibroma MPNST",
]


def fetch_pubmed(pmids: list[str]) -> dict[str, dict[str, str]]:
    if not pmids:
        return {}

    params = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": "synthetic_neurofibroma_literature_builder",
        }
    )
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    with urllib.request.urlopen(url, timeout=90) as response:
        xml_bytes = response.read()

    root = ET.fromstring(xml_bytes)
    records = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="").strip()
        records[pmid] = parse_pubmed_article(article)
    return records


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value


def element_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return clean_text("".join(elem.itertext()))


def parse_year(article: ET.Element) -> str:
    for path in [
        ".//ArticleDate/Year",
        ".//JournalIssue/PubDate/Year",
        ".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year",
    ]:
        year = article.findtext(path)
        if year:
            return year
    medline = article.findtext(".//JournalIssue/PubDate/MedlineDate", default="")
    match = re.search(r"(19|20)\d{2}", medline)
    return match.group(0) if match else ""


def parse_authors(article: ET.Element) -> str:
    authors = []
    for author in article.findall(".//AuthorList/Author"):
        collective = author.findtext("CollectiveName")
        if collective:
            authors.append(clean_text(collective))
            continue
        last = author.findtext("LastName", default="")
        initials = author.findtext("Initials", default="")
        if last:
            authors.append(clean_text(f"{last} {initials}"))
    if len(authors) > 8:
        return "; ".join(authors[:8]) + "; et al."
    return "; ".join(authors)


def parse_abstract(article: ET.Element) -> str:
    parts = []
    for abs_text in article.findall(".//Abstract/AbstractText"):
        label = abs_text.attrib.get("Label")
        text = element_text(abs_text)
        if not text:
            continue
        if label:
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)
    return clean_text(" ".join(parts))


def article_id(article: ET.Element, id_type: str) -> str:
    for elem in article.findall(".//ArticleIdList/ArticleId"):
        if elem.attrib.get("IdType") == id_type:
            return clean_text(elem.text or "")
    return ""


def parse_pubmed_article(article: ET.Element) -> dict[str, str]:
    journal_parts = [
        article.findtext(".//Journal/Title", default=""),
        article.findtext(".//Journal/ISOAbbreviation", default=""),
    ]
    journal = clean_text(journal_parts[0] or journal_parts[1])
    pub_types = [
        element_text(pt)
        for pt in article.findall(".//PublicationTypeList/PublicationType")
        if element_text(pt)
    ]
    return {
        "pmid": article.findtext(".//PMID", default="").strip(),
        "pmcid": article_id(article, "pmc"),
        "doi": article_id(article, "doi"),
        "title": element_text(article.find(".//ArticleTitle")),
        "authors": parse_authors(article),
        "journal": journal,
        "year": parse_year(article),
        "publication_types": "; ".join(pub_types),
        "abstract": parse_abstract(article),
    }


def merge_records(pubmed_records: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for seed in CURATED:
        pmid = seed.get("pmid", "")
        fetched = dict(pubmed_records.get(pmid, {}))
        row = {
            "record_id": pmid or seed.get("doi", ""),
            "pmid": pmid,
            "pmcid": fetched.get("pmcid", ""),
            "doi": seed.get("doi") or fetched.get("doi", ""),
            "title": fetched.get("title") or seed.get("manual_title", ""),
            "authors": fetched.get("authors") or seed.get("manual_authors", ""),
            "journal": fetched.get("journal") or seed.get("manual_journal", ""),
            "year": fetched.get("year") or seed.get("manual_year", ""),
            "url": seed.get("manual_url") or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""),
            "category": seed["category"],
            "category_label": CATEGORY_LABELS[seed["category"]],
            "evidence_type": seed["evidence_type"],
            "key_findings": seed["key_findings"],
            "physical_model_use": seed["physical_model_use"],
            "spatial_scale": seed["spatial_scale"],
            "model_readiness": seed["model_readiness"],
            "publication_types": fetched.get("publication_types", ""),
            "abstract": fetched.get("abstract", ""),
        }
        rows.append(row)
    return rows


def write_json_csv(rows: list[dict[str, str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    json_path = DATA_DIR / "neurofibroma_literature.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = DATA_DIR / "neurofibroma_literature.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bib_key(row: dict[str, str]) -> str:
    first_author = "Unknown"
    if row["authors"]:
        first_author = re.sub(r"[^A-Za-z0-9]", "", row["authors"].split(";")[0].split()[0])
    year = row.get("year") or "nd"
    title_word = "paper"
    title_words = re.findall(r"[A-Za-z0-9]+", row.get("title", ""))
    if title_words:
        title_word = title_words[0]
    return f"{first_author}{year}{title_word}"


def tex_escape(value: str) -> str:
    return (value or "").replace("{", "\\{").replace("}", "\\}")


def write_bibtex(rows: list[dict[str, str]]) -> None:
    entries = []
    for row in rows:
        authors = row["authors"].replace("; ", " and ")
        fields = {
            "title": row["title"],
            "author": authors,
            "journal": row["journal"],
            "year": row["year"],
            "doi": row["doi"],
            "pmid": row["pmid"],
            "pmcid": row["pmcid"],
            "url": row["url"],
        }
        body = "\n".join(
            f"  {key} = {{{tex_escape(value)}}},"
            for key, value in fields.items()
            if value
        )
        entries.append(f"@article{{{bib_key(row)},\n{body}\n}}")
    (DATA_DIR / "neurofibroma_literature.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")


def citation(row: dict[str, str]) -> str:
    id_bits = []
    if row["doi"]:
        id_bits.append(f"DOI: {row['doi']}")
    if row["pmid"]:
        id_bits.append(f"PMID: {row['pmid']}")
    if row["pmcid"]:
        id_bits.append(f"PMCID: {row['pmcid']}")
    id_text = "; ".join(id_bits)
    return f"{row['authors']} ({row['year']}). {row['title']} {row['journal']}. {id_text}. {row['url']}"


def write_annotations(rows: list[dict[str, str]]) -> None:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row)

    chunks = [
        "# Curated Neurofibroma Literature Annotations",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "Scope: NF1-associated neurofibroma, with emphasis on origin, anatomical growth/spread, microenvironment, and physical/model-system evidence.",
        "",
    ]

    for category, label in CATEGORY_LABELS.items():
        category_rows = sorted(grouped.get(category, []), key=lambda r: (r.get("year", ""), r["title"]))
        if not category_rows:
            continue
        chunks.extend([f"## {label}", ""])
        for row in category_rows:
            chunks.extend(
                [
                    f"### {row['title']} ({row['year']})",
                    "",
                    f"- Citation: {citation(row)}",
                    f"- Evidence type: {row['evidence_type']}",
                    f"- Main point: {row['key_findings']}",
                    f"- Modeling use: {row['physical_model_use']}",
                    "",
                ]
            )
    (DATA_DIR / "annotated_literature_review.md").write_text("\n".join(chunks), encoding="utf-8")


def write_search_manifest(rows: list[dict[str, str]]) -> None:
    manifest = {
        "generated": date.today().isoformat(),
        "scope": "NF1-associated neurofibroma literature for origin, growth/spread, microenvironment, and physical/modeling evidence.",
        "queries": SEARCH_QUERIES,
        "source_apis": ["NCBI PubMed E-utilities", "manual DOI/URL entry for newly published mechanobiology paper"],
        "record_count": len(rows),
        "categories": CATEGORY_LABELS,
        "notes": [
            "No PDF copies are stored here; links, identifiers, abstracts, and modeling annotations are stored to avoid copyright problems.",
            "The 2026 mechanical-stiffening paper is included by DOI and publisher URL because PubMed metadata may lag publisher indexing.",
        ],
    }
    (DATA_DIR / "search_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_synthesis(rows: list[dict[str, str]]) -> None:
    by_pmid = {row["pmid"]: row for row in rows if row["pmid"]}
    by_doi = {row["doi"]: row for row in rows if row["doi"]}

    def ref(identifier: str) -> str:
        row = by_pmid.get(identifier) or by_doi.get(identifier)
        if not row:
            return identifier
        label = row["pmid"] or row["doi"]
        return f"[{row['title']}]({row['url']}) ({label})"

    text = f"""# Neurofibroma Growth and Physical Modeling Synthesis

Generated: {date.today().isoformat()}

NF here means NF1-associated neurofibroma unless stated otherwise.

## High-level answer

Neurofibroma growth is best modeled as a Schwann-lineage tumor process that is constrained by nerve anatomy and amplified by a living microenvironment. Plexiform neurofibromas are not usually described as "spreading" by metastasis. They expand locally along nerves, nerve roots, and nerve plexuses, with morbidity coming from local mass effect, infiltration around nerve fascicles, vascular/soft-tissue involvement, and occasional malignant transformation to MPNST.

## Where it is from

The genetic starting point is loss of neurofibromin function from NF1, a tumor-suppressor/RASopathy gene. Mechanistically, this supports modeling NF1 loss as altered RAS/MAPK-centered signaling rather than as a generic proliferation rate: {ref('25877329')} and {ref('26860753')}.

The initiating tumor compartment is Schwann lineage. Foundational mouse genetics tied neurofibroma formation to NF1 loss in Schwann cells plus a permissive environment: {ref('11988578')}. Later work separated subtype-specific origins. Dermal/cutaneous tumors involve susceptible skin-derived precursor or Schwann-lineage populations: {ref('19427294')}. Plexiform neurofibromas are strongly linked to embryonic Schwann-lineage cells in nerve roots and developmental windows of vulnerability: {ref('21551250')} and {ref('25446898')}. Timing and location of NF1 loss matter for subtype: {ref('30348677')}.

## How it grows

The most useful clinical growth data come from volumetric MRI and whole-body MRI cohorts. Growth is heterogeneous and age-dependent. Pediatric and younger patients tend to show faster plexiform neurofibroma growth than adults, while adult internal neurofibromas are often slower or stable over long follow-up: {ref('17215493')}, {ref('23035791')}, and {ref('36332985')}. Whole-body MRI provides tumor-burden observables for calibration: {ref('18559970')}. Growth is also clinically meaningful because volume change can relate to morbidity: {ref('29718344')}.

Distinct nodular lesions and atypical neurofibromas should not be collapsed into the same state as diffuse benign plexiform mass. Longitudinal imaging separates nodular lesion behavior from broader PN growth: {ref('32152628')}. Atypical neurofibromas are premalignant intermediates, with CDKN2A/B deletion and other genomic/pathologic changes marking progression risk: {ref('21987445')} and {ref('29409029')}. Overall benign tumor burden is also associated with MPNST risk: {ref('24166582')}.

## What drives local expansion

The microenvironment is not optional. Nf1+/- bone marrow and c-kit-dependent components are required in classic models: {ref('18984156')}. Mast cells, macrophages, fibroblasts, blood vessels, neurons, and ECM contribute signals and structure: {ref('20233971')}, {ref('16835260')}, {ref('23099891')}, and {ref('29596064')}. Neuronal activity is also a plausible growth input: peripheral neuron activity-regulated COL1A2 increased proliferation of NF1-deficient Schwann cells in NF1 models: {ref('35589737')}.

## Physical modeling implications

A physical model should not be a simple radially expanding sphere unless the goal is only a toy baseline. The literature supports a hybrid or agent-based model on a nerve/plexus graph embedded in tissue:

1. NF1-deficient Schwann-lineage tumor cells as the initiating and proliferating agents.
2. Nf1+/- fibroblasts, mast cells, macrophages, endothelial cells, and neurons as active microenvironment agents or fields.
3. ECM as both a biochemical and mechanical field, with collagen/basement membrane composition and stiffness affecting growth, invasion, and MEK-inhibitor response.
4. Anisotropic growth along nerve fascicles/branches rather than isotropic free-space growth.
5. MRI-observed tumor volumes as patient-scale calibration targets, and 3D cultures/organoids as local parameter-calibration systems.

ECM and mechanics have direct supporting evidence. Single-cell matrisome work maps matrix components and matrix-producing cells in cutaneous neurofibromas: {ref('33413690')}. Basement-membrane ECM proteins relate to development and MEK response: {ref('37140985')}. 3D plexiform neurofibroma models and fibroblast secretome experiments show that spheroid growth and invasiveness can be studied in controlled matrices: {ref('29055717')} and {ref('39061138')}. The most direct mechanobiology paper in this package reports that ECM stiffening promotes growth/invasion phenotypes and reduced selumetinib sensitivity in 3D plexiform neurofibroma cultures: {ref('10.3390/cells15100877')}.

## Model systems available

Useful physical/experimental systems include immortalized normal and NF1 neurofibroma Schwann cells ({ref('27617404')}), pharmacogenomic profiling resources ({ref('29893754')}), 3D plexiform culture systems ({ref('29055717')}), iPSC and humanized neurofibroma models ({ref('30713041')}, {ref('33108355')}, {ref('35172160')}), and patient-derived cutaneous neurofibroma organoids ({ref('38744290')}). For cutaneous neurofibroma model selection, see {ref('37330719')}.

## Biggest gaps

The literature has strong pieces but not a single definitive physical simulator. The biggest missing calibration data are patient-specific tissue mechanics, serial ECM composition, local nerve topology at tumor initiation, cell-density maps through time, and matched MRI-to-histology datasets. A practical first model should therefore use MRI growth rates for macro-scale validation and 3D culture/organoid data for local rules.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def write_modeling_notes(rows: list[dict[str, str]]) -> None:
    text = """# Physical Modeling Notes

These notes translate the literature package into model components.

| Component | Representation | Data to use | Key sources |
|---|---|---|---|
| NF1-deficient Schwann lineage | Initiating/proliferating agents with Ras/MAPK-active state | Cell-origin studies, iPSC models | 11988578, 21551250, 25446898, 30348677, 33108355 |
| Nerve topology | Anisotropic graph/tube scaffold, not isotropic free space | MRI segmentation, anatomical nerve roots/plexus geometry | 17215493, 23035791, 18559970 |
| Growth kinetics | Tumor-specific growth-rate distribution with age dependence | Volumetric MRI and WBMRI cohorts | 17215493, 23035791, 36332985, 39497113 |
| Fibroblasts | Nf1+/- stromal agents secreting growth and ECM-modifying factors | 3D spheroid secretome and microenvironment studies | 16835260, 39061138 |
| Mast cells and macrophages | Recruited immune/stromal agents or density fields | c-kit/mast cell and macrophage inhibition studies | 18984156, 20233971, 23099891, 29596064 |
| ECM composition | Collagen/basement-membrane/hyaluronan fields | Matrisome and ECM-response papers | 33413690, 37140985 |
| ECM mechanics | Stiffness field coupled to proliferation, invasion, and therapy response | 3D matrix stiffness experiments | 10.3390/cells15100877 |
| Neuronal activity | Local nerve activity/paracrine factor field | NF1 neuronal hyperexcitability and COL1A2 work | 35589737 |
| Malignant transition | Optional state change through atypical neurofibroma/nodular lesion | Pathology/genomic cohorts and tumor burden risk | 24166582, 21987445, 29409029, 28592921 |

Recommended first-pass model:

1. Use a nerve-graph scaffold with local tissue radius.
2. Seed NF1-deficient Schwann-lineage cells on susceptible nerve-root/branch regions.
3. Let tumor-cell proliferation depend on intrinsic NF1/RAS state, age/developmental factor, local fibroblast/immune support, ECM stiffness, and nerve activity.
4. Let ECM deposition and stiffness increase through fibroblast/mast-cell/macrophage signals.
5. Calibrate macro growth to volumetric MRI cohorts and micro growth to 3D culture/organoid assays.
"""
    (DATA_DIR / "physical_modeling_notes.md").write_text(text, encoding="utf-8")


def make_plotly_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for category, group in df.groupby("category"):
        color = CATEGORY_COLORS[category]
        fig.add_trace(
            go.Scatter3d(
                x=group["year_numeric"],
                y=group["spatial_scale"],
                z=group["model_readiness"],
                mode="markers+text",
                name=CATEGORY_LABELS[category],
                text=group["short_label"],
                textposition="top center",
                marker={
                    "size": 6 + group["model_readiness"],
                    "color": color,
                    "opacity": 0.9,
                    "line": {"width": 0.5, "color": "white"},
                },
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Year: %{x}<br>"
                    "Category: %{customdata[1]}<br>"
                    "Evidence: %{customdata[2]}<br>"
                    "Model use: %{customdata[3]}<extra></extra>"
                ),
                customdata=group[["title", "category_label", "evidence_type", "physical_model_use"]],
            )
        )
    fig.update_layout(
        title="NF1 Neurofibroma Literature Map: Biology to Physical Modeling",
        scene={
            "xaxis_title": "Publication year",
            "yaxis_title": "Spatial scale: molecular -> patient",
            "zaxis_title": "Model readiness",
            "camera": {"eye": {"x": 1.6, "y": 1.8, "z": 1.1}},
        },
        legend_title="Literature bucket",
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        template="plotly_white",
    )
    return fig


def write_notebook(rows: list[dict[str, str]]) -> None:
    df = pd.DataFrame(rows)
    df["year_numeric"] = pd.to_numeric(df["year"], errors="coerce")
    fallback_year = int(df["year_numeric"].dropna().median())
    df["year_numeric"] = df["year_numeric"].fillna(fallback_year).astype(int)
    df["short_label"] = df.apply(
        lambda row: f"{row['year']} {row['category_label'].split('/')[0][:10]}", axis=1
    )
    fig = make_plotly_figure(df)
    fig_json = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# NF1 Neurofibroma Literature Map\n\n"
            "This notebook is generated by `literature/scripts/build_literature_assets.py`. "
            "The Plotly 3D scatter is interactive in Jupyter: rotate, pan, zoom, and hover points."
        ),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "from pathlib import Path\n\n"
            "data_path = Path('../data/neurofibroma_literature.csv')\n"
            "df = pd.read_csv(data_path)\n"
            "df[['year', 'category_label', 'evidence_type', 'title']].head()",
            execution_count=1,
            outputs=[
                nbf.v4.new_output(
                    "execute_result",
                    data={"text/plain": df[["year", "category_label", "evidence_type", "title"]].head().to_string(index=False)},
                    execution_count=1,
                )
            ],
        ),
        nbf.v4.new_code_cell(
            "import plotly.graph_objects as go\n\n"
            "# Axes: x = year, y = biological spatial scale, z = readiness for physical modeling.\n"
            "fig",
            execution_count=2,
            outputs=[
                nbf.v4.new_output(
                    "display_data",
                    data={
                        "application/vnd.plotly.v1+json": fig_json,
                        "text/plain": "Interactive Plotly 3D literature map",
                    },
                    metadata={},
                )
            ],
        ),
    ]
    (VIZ_DIR / "literature_map.ipynb").write_text(nbf.writes(nb), encoding="utf-8")


def write_gif(rows: list[dict[str, str]]) -> None:
    df = pd.DataFrame(rows)
    df["year_numeric"] = pd.to_numeric(df["year"], errors="coerce")
    df["year_numeric"] = df["year_numeric"].fillna(df["year_numeric"].median()).astype(int)
    frames = []

    x_min, x_max = df["year_numeric"].min() - 1, df["year_numeric"].max() + 1
    y_min, y_max = 0, 4.7
    z_min, z_max = 2.5, 5.3

    for angle in range(20, 380, 12):
        fig = plt.figure(figsize=(8, 6), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        for category, group in df.groupby("category"):
            ax.scatter(
                group["year_numeric"],
                group["spatial_scale"],
                group["model_readiness"],
                s=45 + group["model_readiness"] * 15,
                color=CATEGORY_COLORS[category],
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("Year")
        ax.set_ylabel("Spatial scale")
        ax.set_zlabel("Model readiness")
        ax.set_title("NF1 Neurofibroma Literature Map")
        ax.view_init(elev=24, azim=angle)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=CATEGORY_COLORS[key], markersize=7)
            for key, label in CATEGORY_LABELS.items()
        ]
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(-0.04, 1.02), fontsize=7)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

    imageio.mimsave(VIZ_DIR / "literature_map_rotation.gif", frames, duration=0.12)


def write_readme_summary(rows: list[dict[str, str]]) -> None:
    counts = pd.DataFrame(rows)["category_label"].value_counts().to_dict()
    counts_text = "\n".join(f"- {category}: {count}" for category, count in counts.items())
    text = f"""# Literature Package

This folder contains a curated literature set for NF1-associated neurofibroma origin, local growth/spread, microenvironment, and physical/model-system evidence.

## Contents

- `data/neurofibroma_literature.csv`: structured paper table with PubMed/DOI identifiers and modeling annotations.
- `data/neurofibroma_literature.json`: JSON version of the same dataset.
- `data/neurofibroma_literature.bib`: BibTeX citations.
- `data/annotated_literature_review.md`: per-paper annotations.
- `data/physical_modeling_notes.md`: translation from literature to model components.
- `data/search_manifest.json`: search queries, scope, and generation metadata.
- `visualizations/literature_map_rotation.gif`: rotating 3D literature-map GIF.
- `visualizations/literature_map.ipynb`: generated notebook with an interactive Plotly 3D map.

## Category Counts

{counts_text}

No HTML visualization files are generated.
"""
    (ROOT / "PACKAGE_README.md").write_text(text, encoding="utf-8")


def validate_no_html_files() -> None:
    html_files = list(VIZ_DIR.glob("*.html"))
    if html_files:
        raise RuntimeError(f"Unexpected HTML files in visualizations: {html_files}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    pmids = [seed["pmid"] for seed in CURATED if seed.get("pmid")]
    pubmed_records = fetch_pubmed(pmids)
    rows = merge_records(pubmed_records)

    write_json_csv(rows)
    write_bibtex(rows)
    write_annotations(rows)
    write_search_manifest(rows)
    write_synthesis(rows)
    write_modeling_notes(rows)
    write_notebook(rows)
    write_gif(rows)
    write_readme_summary(rows)
    validate_no_html_files()

    print(f"Wrote {len(rows)} literature records under {ROOT}")
    print(f"Data: {DATA_DIR}")
    print(f"Visualizations: {VIZ_DIR}")


if __name__ == "__main__":
    main()
