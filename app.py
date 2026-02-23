"""
Streamlit frontend for the AI School Recommendation system.
"""

import streamlit as st
from backend import (
    SCORE_PG_RULES,
    VALID_CCA_TYPES,
    VALID_POSTING_GROUPS,
    VALID_SCH_TYPES,
    VALID_ZONES,
    build_rag_chain,
    # debug_logs,
    get_cca_groups,
    get_locations,
    init_vectordb,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI School Recommender",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        color: #1a5276;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #5d6d7e;
        font-size: 1.05rem;
    }
    .stSlider > div > div > div > div {
        background-color: #1a5276;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #d5dbdb;
        border-radius: 8px;
    }
    .score-info {
        background-color: rgba(41, 128, 185, 0.15);
        border-left: 4px solid #2980b9;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>AI Secondary School Recommender</h1>
    <p>Find the best-fit Singapore secondary schools based on your PSLE score, location, and CCA interests</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading school database...")
def load_backend():
    vectordb, _ = init_vectordb()
    rag_chain = build_rag_chain(vectordb)
    return rag_chain


@st.cache_data(show_spinner=False)
def load_cca_groups():
    return get_cca_groups()


@st.cache_data(show_spinner=False)
def load_locations():
    return get_locations()


rag_chain = load_backend()
cca_groups = load_cca_groups()
locations = load_locations()


# ---------------------------------------------------------------------------
# Helper: get valid posting groups for a given score
# ---------------------------------------------------------------------------
def get_valid_pgs(score: int) -> list[str]:
    """Return posting groups valid for the given PSLE score."""
    valid = []
    for (lo, hi), groups in SCORE_PG_RULES.items():
        if lo <= score <= hi:
            valid = list(groups) 
            break
    if score < 9: valid.append("ip")
    return valid


PG_DISPLAY = {
    "ip": "IP (Integrated Programme)",
    "pg3": "PG3 (Posting Group 3)",
    "pg2": "PG2 (Posting Group 2)",
    "pg1": "PG1 (Posting Group 1)",
}

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
col_form, col_result = st.columns([1, 2], gap="large")

with col_form:
    st.subheader("Student Profile")

    # -- PSLE Score --
    user_score = st.slider(
        "PSLE Score",
        min_value=4, max_value=30, value=10, step=1,
        help="Lower score = better performance. Range: 4 (best) to 30.",
    )

    valid_pgs = get_valid_pgs(user_score)
    pg_options = {PG_DISPLAY.get(pg, pg.upper()): pg for pg in valid_pgs}

    # st.markdown(
    #     f'<div class="score-info">Score <strong>{user_score}</strong> qualifies for: '
    #     f'<strong>{", ".join(pg.upper() for pg in valid_pgs if pg != "ip")}</strong>'
    #     f' (and IP if COP allows)</div>',
    #     unsafe_allow_html=True,
    # )

    # -- Posting Group --
    pg_label = st.selectbox(
        "Posting Group",
        options=list(pg_options.keys()),
        help="Select based on your PSLE score range.",
    )
    posting_group = pg_options[pg_label]

    # -- School Type --
    sch_type_options = st.multiselect(
        "School Type",
        options=VALID_SCH_TYPES,
        default=VALID_SCH_TYPES,
        help="Select one or more school types.",
    )

    st.divider()
    st.subheader("Preferences")

    # -- Zone & Location --
    user_zone = st.multiselect(
        "Preferred Zone(s)",
        options=VALID_ZONES,
        help="Leave empty for no zone preference.",
    )

    user_location = st.multiselect(
        "Preferred Location(s)",
        options=locations,
        help="Select one or more districts. Leave empty for no preference.",
    )

    st.divider()

    # -- CCA --
    user_cca_type = st.multiselect(
        "CCA Category",
        options=VALID_CCA_TYPES,
        help="Select one or more CCA categories. Leave empty for no preference.",
    )

    user_cca_grp = st.text_input(
        "Specific CCA",
        placeholder="e.g. Robotics, Basketball, Choir",
        help="Type one or more CCAs separated by commas. Leave blank for no preference.",
    )

    st.divider()

    # -- Weights --
    st.markdown("**Priority Weights** (must total 1.0)")
    w_loc = st.slider("Location", 0.0, 1.0, 0.5, 0.1)
    w_cca = round(1.0 - w_loc, 1)
    st.write(f"CCA: **{w_cca}**")

    st.divider()

    # -- Submit --
    submitted = st.button("Get Recommendations", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Build input & run
# ---------------------------------------------------------------------------
with col_result:
    if submitted:
        if not sch_type_options:
            st.error("Please select at least one school type.")
        else:
            loc_str = ", ".join(user_location) if user_location else None
            cca_type_val = user_cca_type if user_cca_type else None
            cca_grp_val = user_cca_grp.strip() if user_cca_grp.strip() else None

            user_input = {
                "user_score": user_score,
                "posting_group": posting_group,
                "user_sch_type": sch_type_options,
                "w_loc": w_loc,
                "w_cca": w_cca,
                "user_zone": user_zone if user_zone else None,
                "user_location": loc_str,
                "user_cca_type": cca_type_val,
                "user_cca_grp": cca_grp_val,
            }

            # with st.spinner("Searching schools and fetching travel info..."):
            #     try:
            #         response = rag_chain.invoke(user_input)
            #     except Exception as e:
            #         response = None
            #         st.error(f"An error occurred: {str(e)}")

            with st.status("Finding the best schools for you...", expanded=True) as status:
                try:
                    st.write("Searching schools and fetching travel info...")
                    # Your chain is invoked here
                    response = rag_chain.invoke(user_input)
                    
                    status.update(label="Recommendations ready!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    status.update(label="Search failed.", state="error")
                    response = None

            if response:
                st.subheader("Recommended Schools")
                with st.container(height=600):
                    st.markdown(response)
            elif response == "":
                st.warning("No schools matched your criteria. Try widening your search distance or score range.")
    
            # with st.expander("Debug Info", expanded=False):
            #     st.markdown(f"**Input:** {user_input}")
            #     if debug_logs:
            #         for log in debug_logs:
            #             st.markdown(log)
            #     else:
            #         st.write("No debug logs captured.")
    else:
        st.markdown(
            """
            ### How to use
            1. Set your **PSLE score** and **posting group** on the left

            | Posting Group | Score Range |
            | :--- | :--- |
            | **IP** | 4 - 8 |
            | **PG3** | 4 - 20 | 
            | **PG2** | 21 - 25 |
            | **PG1** | 26 - 30 |
            
            2. Choose your preferred **zone**, **location**, and **CCA**
            3. Adjust the **priority weights** between location and CCA
            4. Click **Get Recommendations**

            The system will retrieve matching schools from the database,
            rank them based on your preferences, search for travel information,
            and generate a personalised recommendation using AI.
            """
        )
