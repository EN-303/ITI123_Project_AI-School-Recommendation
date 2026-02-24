"""
Backend module for the AI School Recommendation system.
"""

import os
import re
import shutil
import unicodedata

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from serpapi import GoogleSearch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
import gdown

def _get_secret(key: str) -> str | None:
    """Retrieve a secret from Streamlit Cloud secrets or environment variables."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


def _download_assets():
    """Download data files and vector DB from Google Drive if not present."""
    DATA_FOLDER_URL = "https://drive.google.com/drive/folders/1JKEuuseIP9qQjp0NJqerykv3ZBvwpPub?usp=drive_link"
    if not os.path.exists("data"):
        gdown.download_folder(url=DATA_FOLDER_URL, quiet=False, use_cookies=False)

    VECTORDB_FOLDER_URL = "https://drive.google.com/drive/folders/1YKNgoWTqnIpzdLKbS8FUj4ceqvpIcFhl?usp=drive_link"
    if not os.path.exists("sch_vector_db"):
        gdown.download_folder(url=VECTORDB_FOLDER_URL, quiet=False, use_cookies=False)

_download_assets()

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
SERPAPI_KEY = _get_secret("SERPAPI_API_KEY")

EMBED_MODEL = "text-embedding-3-large"
PERSIST_DIR = "./sch_vector_db"

S_COP = "data/COP.csv"
S_SCHINFO = "data/schinfo.csv"
S_CCA = "data/CCA.csv"

# Valid option lists (used by both backend validation and frontend dropdowns)
VALID_POSTING_GROUPS = ["ip", "pg3", "pg2", "pg1"]
VALID_SCH_TYPES = ["CO-ED SCHOOL", "BOYS' SCHOOL", "GIRLS' SCHOOL"]
VALID_ZONES = ["NORTH", "SOUTH", "EAST", "WEST"]
VALID_CCA_TYPES = [
    "PHYSICAL SPORTS",
    "CLUBS AND SOCIETIES",
    "VISUAL AND PERFORMING ARTS",
    "UNIFORMED GROUPS",
    "OTHERS",
]

SCORE_PG_RULES = {
    (4, 20): ["pg3"],
    (21, 22): ["pg3", "pg2"],
    (23, 24): ["pg2"],
    (25, 25): ["pg2", "pg1"],
    (26, 30): ["pg1"],
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def filter_secondary_only(df, colname):
    return df[~df[colname].str.contains("PRIMARY|JUNIOR|CENTRALISED", case=False, na=False)]


# ---------------------------------------------------------------------------
# Initialisation (called once via st.cache_resource)
# ---------------------------------------------------------------------------

def init_vectordb():
    """Load or rebuild the ChromaDB vector store. Returns (vectordb, all_docs)."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set. Please add it to your .env file.")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

    try:
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        vectordb.get(limit=1)
    except Exception:
        raise RuntimeError(f"Failed to initialize ChromaDB: {str(e)}")

    return vectordb

def get_cca_groups():
    """Return sorted list of unique CCA grouping descriptions."""
    df = pd.read_csv(S_CCA)
    df = filter_secondary_only(df, "school_section")
    return sorted(df["cca_grouping_desc"].dropna().unique().tolist())


def get_locations():
    """Return sorted list of unique location (dgp_code) values."""
    df = pd.read_csv(S_SCHINFO)
    df = filter_secondary_only(df, "mainlevel_code")
    return sorted(df["dgp_code"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def sch_retriever(vectordb, user_score: int, posting_group: str, user_sch_type=None, k: int = 40):
    s_cop = S_COP
    s_schinfo = S_SCHINFO

    schinfo_conditions = [{"source": {"$eq": s_schinfo}}]
    if not user_sch_type:
        user_sch_type = ["CO-ED SCHOOL", "BOYS' SCHOOL", "GIRLS' SCHOOL"]
    type_list = user_sch_type if isinstance(user_sch_type, list) else [user_sch_type]
    schinfo_conditions.append({"nature_code": {"$in": type_list}})

    try:
        sch_results = vectordb.get(where={"$and": schinfo_conditions})
    except Exception:
        return []
    if not sch_results.get("metadatas"):
        return []

    sch_keys = [meta["school_key"] for meta in sch_results["metadatas"]]
    if not sch_keys:
        return []

    cop_field = f"{posting_group}_cop"
    if posting_group == "ip":
        # IP posting group: filter only on ip_cop
        cop_filter = {
            "$and": [
                {"source": {"$eq": s_cop}},
                {"school_key": {"$in": sch_keys}},
                {"ip_cop": {"$gte": user_score}},
                {"ip_cop": {"$lt": 99}}
            ]
        }
    else:
        # Non-IP posting groups (PG3/PG2/PG1): filter only on the selected PG's COP
        # Excludes IP-only schools
        cop_filter = {
            "$and": [
                {"source": {"$eq": s_cop}},
                {"school_key": {"$in": sch_keys}},
                {cop_field: {"$gte": user_score}},
                {cop_field: {"$lt": 99}}
            ]
        }

    try:
        all_schs = vectordb.get(where=cop_filter, limit=k)
    except Exception:
        return []

    if not all_schs.get('ids') or len(all_schs['ids']) == 0:
        return []

    eligible_list = []
    for i in range(len(all_schs["ids"])):
        meta = all_schs["metadatas"][i]
        sch_cop = int(meta.get(cop_field, 0))
        meta["cop_gap"] = abs(user_score - sch_cop)
        eligible_list.append(Document(page_content=all_schs["documents"][i], metadata=meta))

    eligible_list.sort(key=lambda x: x.metadata["cop_gap"])
    return eligible_list[:k]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def weighted_sch_ranker(vectordb, eligible_schs, w_loc, w_cca,
                        user_zone=None, user_location=None,
                        user_cca_type=None, user_cca_grp=None):
    s_schinfo = S_SCHINFO
    s_cca = S_CCA

    eligible_keys = [doc.metadata.get("school_key") for doc in eligible_schs]
    if not eligible_keys:
        return []

    school_scores = {}

    try:
        loc_results = vectordb.get(
            where={"$and": [{"school_key": {"$in": eligible_keys}}, {"source": {"$eq": s_schinfo}}]}
        )
    except Exception:
        return []
    if not loc_results.get('ids') or len(loc_results['ids']) == 0:
        return []

    cca_batch_filter = {"$and": [{"school_key": {"$in": eligible_keys}}, {"source": {"$eq": s_cca}}]}
    if user_cca_type:
        type_list = user_cca_type if isinstance(user_cca_type, list) else [user_cca_type]
        cca_batch_filter["$and"].append({"cca_type": {"$in": type_list}})
    try:
        all_cca_data = vectordb.get(where=cca_batch_filter, limit=5000)
    except Exception:
        all_cca_data = {"ids": [], "metadatas": [], "documents": []}

    cca_by_school = {}
    for cca_idx in range(len(all_cca_data.get("ids", []))):
        cca_key = all_cca_data["metadatas"][cca_idx].get("school_key")
        if cca_key not in cca_by_school:
            cca_by_school[cca_key] = []
        cca_by_school[cca_key].append({
            "page_content": all_cca_data["documents"][cca_idx],
            "metadata": all_cca_data["metadatas"][cca_idx],
        })

    # Map location metadata to school_scores
    for i in range(len(loc_results["ids"])):
        meta = loc_results["metadatas"][i]
        key = meta.get("school_key")
        if key not in school_scores:
            school_scores[key] = {
                "loc_score": 0, "cca_score": 0,
                "sch_info": meta, "sch_content": loc_results["documents"][i],
                "cca_info": None,
            }

        zone_match = False
        if user_zone:
            zones = user_zone if isinstance(user_zone, list) else [user_zone]
            if meta.get("zone_code", "").lower() in [z.lower() for z in zones]:
                zone_match = True

        dgp_match = False
        if user_location:
            location_terms = [loc.strip().lower() for loc in user_location.split(",") if loc.strip()]
            dgp_code_lower = meta.get("dgp_code", "").lower()
            dgp_match = any(loc in dgp_code_lower for loc in location_terms)

        # Check location in Metadata (handle comma-separated values)
        dgp_match = False
        if user_location:
            location_terms = [loc.strip().lower() for loc in user_location.split(',') if loc.strip()]
            dgp_code_lower = meta.get('dgp_code', '').lower()
            dgp_match = any(loc in dgp_code_lower for loc in location_terms)

        # If schinfo docs matches, set score to 1
        if zone_match:
            school_scores[key]["loc_score"] = 1

        if zone_match and dgp_match:
            school_scores[key]["loc_score"] = 2

        if not zone_match and dgp_match:
            school_scores[key]["loc_score"] = 1

        # Map CCA =============
        cca_items = cca_by_school.get(key, [])
        cca_match = []
        if cca_items:
            search_terms = []
            if user_cca_grp:
                search_terms = [term.strip().lower() for term in user_cca_grp.split(',') if term.strip()]
            search_variants = set()
            for term in search_terms:
                search_variants.add(term)
                if term.endswith('s'):
                    search_variants.add(term[:-1])
                else:
                    search_variants.add(term + 's')
            for item in cca_items:
                cca_grp_lower = item["metadata"].get("cca_grp", "").lower()
                matched = False
                for v in search_variants:
                    if v in cca_grp_lower:
                        matched = True
                        break
                if matched:
                    from langchain_core.documents import Document
                    cca_match = [Document(page_content=item["page_content"], metadata=item["metadata"])]
                    break
            if not cca_match and user_cca_type:
                from langchain_core.documents import Document
                cca_match = [Document(page_content=cca_items[0]["page_content"], metadata=cca_items[0]["metadata"])]

        if cca_match:
            match_doc = cca_match[0]
            school_scores[key]["cca_info"] = match_doc.metadata

            if not user_cca_grp:
                if user_cca_type:
                    school_scores[key]["cca_score"] = 1
            else:
                meta_grp = match_doc.metadata.get('cca_grp', '').lower()

                cca_terms = [term.strip().lower() for term in user_cca_grp.split(',') if term.strip()]
                score_variants = set()
                for term in cca_terms:
                    score_variants.add(term)
                    if term.endswith('s'):
                        score_variants.add(term[:-1])
                    else:
                        score_variants.add(term + 's')
                cca_matched = any(v in meta_grp for v in score_variants)

                if user_cca_type and cca_matched:
                    school_scores[key]["cca_score"] = 2

                if not user_cca_type and cca_matched:
                    school_scores[key]["cca_score"] = 1

                if user_cca_type and not cca_matched:
                    school_scores[key]["cca_score"] = 1

        #get cop_gap =============
        sch_cop_gap = next((item for item in eligible_schs if item.metadata.get("school_key") == key), None)
        if sch_cop_gap:
            cop_gap = sch_cop_gap.metadata.get("cop_gap", 999)
            cop = (
                f"[IP: {sch_cop_gap.metadata.get('ip_cop', 'N/A')} | "
                f"PG3: {sch_cop_gap.metadata.get('pg3_cop', 'N/A')} | "
                f"PG2: {sch_cop_gap.metadata.get('pg2_cop', 'N/A')} | "
                f"PG1: {sch_cop_gap.metadata.get('pg1_cop', 'N/A')}]"
            )
        else:
            cop_gap = 999
            cop = "[IP: N/A | PG3: N/A | PG2: N/A | PG1: N/A]"

        school_scores[key]["cop_gap"] = cop_gap
        school_scores[key]["cop"] = cop

    # 4. Final Weighted Ranking
    final_list = []
    for key, data in school_scores.items():
        total_score = (data["loc_score"] * w_loc) + (data["cca_score"] * w_cca)
        final_list.append({
            "school": key,
            "school_name": data["sch_info"].get("sch_name", ""),
            "school_url": data["sch_info"].get("sch_url", ""),
            "score": total_score,
            "cop_gap": data.get("cop_gap", 999),
            "cop": data.get("cop", "[IP: N/A | PG3: N/A | PG2: N/A | PG1: N/A]"),
            "location": data["sch_info"].get("dgp_code", ""),
            "matched_cca": data["cca_info"].get("cca_grp", "") if data["cca_info"] else "",
            "sch_metadata": data["sch_info"],
            "sch_content": data.get("sch_content", ""),
            "cca_metadata": data["cca_info"],
        })

    final_list.sort(key=lambda x: (-x["score"], x["cop_gap"]))
    top_6 = final_list[:6]

    print(f"**Top {len(top_6)} Schools**")
    for i, s in enumerate(top_6, 1):
        print(f"  {i}. {s['school_name']} ")
    
    return top_6


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def search_travel_info(school_name: str) -> dict:
    empty_result = {
        "info": "Travel info unavailable",
        "sources": [],
        "knowledge_graph": {}
    }

    if not SERPAPI_KEY:
        empty_result["info"] = "Travel info unavailable (API key not configured)"
        return empty_result

    query = f"{school_name} Singapore nearest MRT station bus routes how to get there"

    try:
        params = {
            "engine": "google",
            "q": query,
            "location": "Singapore",
            "api_key": SERPAPI_KEY,
            "num": 5
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        travel_snippets = []
        sources = []
        kg_details = {}

        # Check answer box for direct answers
        if "answer_box" in results:
            ab = results["answer_box"]
            answer = ab.get("answer") or ab.get("snippet", "")
            if answer:
                travel_snippets.append(answer)
            ab_link = ab.get("link", "")
            ab_title = ab.get("title", "Answer Box")
            if ab_link:
                sources.append({"title": ab_title, "link": ab_link})

        # Extract knowledge graph details
        if "knowledge_graph" in results:
            kg = results["knowledge_graph"]
            for field in ["title", "address", "phone", "type", "description"]:
                if field in kg:
                    kg_details[field] = kg[field]
            if "address" in kg:
                travel_snippets.append(f"Address: {kg['address']}")
            kg_source = kg.get("knowledge_graph_search_link") or kg.get("serpapi_knowledge_graph_search_link", "")
            if kg.get("source"):
                sources.append({"title": kg["source"].get("name", "Knowledge Graph"), "link": kg["source"].get("link", "")})
            elif kg.get("website"):
                sources.append({"title": kg.get("title", "School Website"), "link": kg["website"]})

        # Extract transport-related snippets from organic results with source links
        transport_keywords = [
            "mrt", "bus", "walk", "station", "transport",
            "minute", "km", "distance", "route", "nearest"
        ]
        organic = results.get("organic_results", [])[:5]

        for r in organic:
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            title = r.get("title", "")
            if link:
                sources.append({"title": title, "link": link})
            if snippet and any(kw in snippet.lower() for kw in transport_keywords):
                travel_snippets.append(snippet)
                if len(travel_snippets) >= 3:
                    break

        # Fallback to first available snippets
        if not travel_snippets:
            for r in organic[:2]:
                snippet = r.get("snippet", "")
                if snippet:
                    travel_snippets.append(snippet)

        info = " | ".join(travel_snippets[:3]) if travel_snippets else "No specific travel information found"
        unique_sources = list({s["link"]: s for s in sources if s.get("link")}.values())

        return {
            "info": info,
            "sources": unique_sources[:3],
            "knowledge_graph": kg_details
        }

    except Exception as e:
        empty_result["info"] = f"Web search unavailable: {str(e)}"
        return empty_result

    print(f"SerpAPI key loaded: {'Yes' if SERPAPI_KEY else 'No'}")

# ---------------------------------------------------------------------------
# Full ranking logic (context builder)
# ---------------------------------------------------------------------------

def full_ranking_logic(vectordb, inputs):
    user_score = inputs.get("user_score")
    if user_score is None:
        return "INVALID_INPUT: user_score is required."
    if not isinstance(user_score, (int, float)) or user_score < 4 or user_score > 32:
        return "INVALID_INPUT: user_score must be between 4 and 32."

    posting_group = inputs.get("posting_group")
    if posting_group is None:
        return "INVALID_INPUT: posting_group is required."
    if posting_group.lower() not in VALID_POSTING_GROUPS:
        return f"INVALID_INPUT: posting_group must be one of {VALID_POSTING_GROUPS}."

    pg = posting_group.lower()
    score = int(user_score)

    if pg != "ip":
        allowed = []
        for (lo, hi), groups in SCORE_PG_RULES.items():
            if lo <= score <= hi:
                allowed = groups
                break
        if allowed and pg not in allowed:
            return (
                f"INVALID_INPUT: PSLE score {score} is not compatible with posting group '{pg.upper()}'. "
                f"Valid posting group(s) for score {score}: {', '.join(g.upper() for g in allowed)}. "
                f"Score ranges: 4-20 \u2192 PG3 | 21-22 \u2192 PG3/PG2 | 23-24 \u2192 PG2 | 25 \u2192 PG2/PG1 | 26-30 \u2192 PG1."
            )

    user_sch_type = inputs.get("user_sch_type")
    if user_sch_type:
        sch_types = user_sch_type if isinstance(user_sch_type, list) else [user_sch_type]
        for st in sch_types:
            if st not in VALID_SCH_TYPES:
                return f"INVALID_INPUT: user_sch_type must be one of {VALID_SCH_TYPES}."

    w_loc = inputs.get("w_loc")
    if w_loc is None:
        return "INVALID_INPUT: w_loc is required."
    if not isinstance(w_loc, (int, float)) or w_loc < 0 or w_loc > 1:
        return "INVALID_INPUT: w_loc must be between 0 and 1."

    w_cca = inputs.get("w_cca")
    if w_cca is None:
        return "INVALID_INPUT: w_cca is required."
    if not isinstance(w_cca, (int, float)) or w_cca < 0 or w_cca > 1:
        return "INVALID_INPUT: w_cca must be between 0 and 1."

    if abs((w_loc + w_cca) - 1.0) > 0.001:
        return f"INVALID_INPUT: w_loc ({w_loc}) + w_cca ({w_cca}) must sum to 1.0."

    user_zone = inputs.get("user_zone")
    if user_zone:
        zones = user_zone if isinstance(user_zone, list) else [user_zone]
        for z in zones:
            if z.upper() not in VALID_ZONES:
                return f"INVALID_INPUT: user_zone must be one of {VALID_ZONES}."

    user_cca_type = inputs.get("user_cca_type")
    if user_cca_type:
        cca_types = user_cca_type if isinstance(user_cca_type, list) else [user_cca_type]
        for ct in cca_types:
            if ct.upper() not in VALID_CCA_TYPES:
                return f"INVALID_INPUT: user_cca_type must be one of {VALID_CCA_TYPES}."

    eligible_schs = sch_retriever(vectordb, user_score=user_score, posting_group=pg, user_sch_type=user_sch_type)
    pg_upper = pg.upper()

    if not eligible_schs:
        return (
            f"NO_RESULTS: No schools found for posting group '{pg_upper}' "
            f"with PSLE score {user_score}. "
            f"Try a different posting group that matches the student's score range."
        )

    top_6 = weighted_sch_ranker(
        vectordb, eligible_schs,
        w_loc=w_loc, w_cca=w_cca,
        user_zone=inputs.get("user_zone"),
        user_location=inputs.get("user_location"),
        user_cca_type=inputs.get("user_cca_type"),
        user_cca_grp=inputs.get("user_cca_grp"),
    )

    if not top_6:
        return (
            f"NO_RESULTS: {len(eligible_schs)} schools were eligible based on COP, "
            f"but none matched the ranking criteria. Try adjusting zone, location, or CCA preferences."
        )

    # Include all top 6 recommendations
    if top_6:
        context = f"Student PSLE Score: {user_score}\n"
        context += f"Posting Group: {pg_upper}\n"
        context += "(Schools sorted by Weighted Score descending, then COP Gap ascending)\n\n"
    
        for idx, sch in enumerate(top_6, 1):
            zone = sch["sch_metadata"].get("zone_code", "N/A")
            cop_gap = sch.get("cop_gap", "N/A")
            cop_display = sch["cop"].replace(": 99.0", ": N/A").replace(": 99", ": N/A")
    
            db_ref = {"address": "N/A", "postal_code": "N/A"}
            sch_content = sch.get("sch_content", "")
            if sch_content:
                for line in sch_content.split("\n"):
                    line_lower = line.strip().lower()
                    if line_lower.startswith("address:"):
                        db_ref["address"] = line.split(":", 1)[1].strip()
                    elif line_lower.startswith("postal_code:"):
                        db_ref["postal_code"] = line.split(":", 1)[1].strip()
    
            db_ref_str = (
                f"Address: {db_ref['address']}, S{db_ref['postal_code']}"
            )
    
            cca_meta = sch.get("cca_metadata") or {}
            cca_grp = cca_meta.get("cca_grp", "")
            cca_type = cca_meta.get("cca_type", "")
            cca_ref_str = f"{cca_grp} ({cca_type})" if cca_grp else "None"
    
            travel_result = search_travel_info(sch["school_name"])
            travel_info = travel_result.get("info", "N/A")
            travel_sources = travel_result.get("sources", [])
            travel_kg = travel_result.get("knowledge_graph", {})
    
            kg_str = ""
            if travel_kg:
                kg_parts = [f"{k}: {v}" for k, v in travel_kg.items()]
                kg_str = "; ".join(kg_parts)
    
            sources_str = ""
            if travel_sources:
                src_parts = [f"{s['title']} ({s['link']})" for s in travel_sources]
                sources_str = ", ".join(src_parts)
    
            if sources_str:
                ref_line = f"References: ChromaDB: COP, School Info, CCA | Web: {sources_str}"
            else:
                ref_line = "References: ChromaDB: COP, School Info, CCA"
    
            context += (
                f"Recommendation #{idx}. {sch['school_name']}\n"
                f"  URL: {sch['school_url']}\n"
                f"  Weighted Score: {sch['score']:.2f}\n"
                f"  [ChromaDB - COP] COP: {cop_display} | COP Gap: {cop_gap}\n"
                f"  [ChromaDB - School Info] Zone: {zone} | Location: {sch['location'] or 'None'}\n"
                f"  [ChromaDB - School Info] {db_ref_str}\n"
                f"  [ChromaDB - CCA] Matched CCA: {cca_ref_str}\n"
                f"  [Web Search] Travel Info: {travel_info}\n"
                f"  [Web Search] Knowledge Graph: {kg_str or 'N/A'}\n"
                f"  {ref_line}\n\n"
            )

    return context


# ---------------------------------------------------------------------------
# LLM Chain
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """
You are a Singapore secondary school education consultant helping a student
with PSLE score {user_score} ({posting_group} posting group) find suitable schools.

STRICT RULES:
- ONLY use facts from the MATCHED SCHOOLS data below
- DO NOT add information not present in the data
- DO NOT invent school names — only use schools listed below
- Use exact numbers when stating COP values, scores, and gaps
- DO NOT use subjective words like "excellent", "perfect", "ideal", "strong" — instead state the factual comparison (e.g., "COP gap of 2" not "excellent COP match")

STUDENT PROFILE:
- PSLE Score: {user_score} | Posting Group: {posting_group}
- School Type: {user_sch_type}
- Preferred Zone(s): {user_zone} | Preferred Location(s): {user_location}
- CCA Interest: {user_cca_type} ({user_cca_grp})
- Location Weight: {w_loc} | CCA Weight: {w_cca}

MATCHED SCHOOLS:
{formatted_context}

INSTRUCTIONS:

1. If the data starts with "INVALID_INPUT:", "NO_RESULTS:", or is empty:
   - Explain the issue and suggest adjusting the posting group or preferences
   - Do NOT list any schools

2. Start with: "Based on your PSLE score of {user_score} and your preferences for {user_zone} zone and {user_cca_type} CCA, here are the recommended {posting_group} schools:"

3. You MUST list ALL schools provided in the MATCHED SCHOOLS data above. Do NOT skip or omit any school. If there are 6 schools, list all 6. Every "Recommendation #" entry must appear in your output.

4. For each school, write a recommendation in this format:

   **#[rank]. [SCHOOL NAME]**
   URL: [url from data]

   This school has a {posting_group} COP of [value from data], giving a COP gap of [value from data] from your score of {user_score}. The school is in the [zone from data] zone, located in [location from data][state "which matches" or "which differs from" your preference of {user_zone}/{user_location}]. [If CCA matched: "The school offers [CCA name from data] under [CCA type from data]." If no match: "No matching CCA was found for this school."]. [Use the [Web Search] Travel Info line to describe nearest MRT station and bus routes.]

   Copy the "References:" line from the data EXACTLY as provided — do NOT modify, shorten, or omit any part of it. Include the full line with all URLs.
   References: ChromaDB: COP, School Info, CCA | Web: SchoolName Info - SiteName (https://example.com/page1), Getting There (https://example.com/page2)

5. If zone/location preference is "Any", skip the location comparison.

6. End with one sentence encouraging the student to visit the school websites for more details.

IMPORTANT: You MUST include every single school from the MATCHED SCHOOLS data. Do NOT summarize, merge, or skip any schools. You MUST copy the References line verbatim for EVERY school.
"""


def build_rag_chain(vectordb):
    """Build and return the LCEL RAG chain."""
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=16384)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    output_parser = StrOutputParser()

    def _ranking_wrapper(inputs):
        return full_ranking_logic(vectordb, inputs)

    rag_chain = (
        {
            "formatted_context": RunnableLambda(_ranking_wrapper),
            "user_score": lambda x: x["user_score"],
            "posting_group": lambda x: x.get("posting_group", "N/A").upper(),
            "user_sch_type": lambda x: x.get("user_sch_type") or "Any",
            "user_zone": lambda x: x.get("user_zone") or "Any",
            "user_location": lambda x: x.get("user_location") or "Any",
            "user_cca_type": lambda x: x.get("user_cca_type") or "Any",
            "user_cca_grp": lambda x: x.get("user_cca_grp") or "Any",
            "w_loc": lambda x: x.get("w_loc", 0.5),
            "w_cca": lambda x: x.get("w_cca", 0.5),
        }
        | prompt
        | llm
        | output_parser
    )
    return rag_chain
