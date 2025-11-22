import streamlit as st
import google.generativeai as genai
import os

# ----------------------------------------------------
# 1. Load Gemini API key from Streamlit Secrets
# ----------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🚨 GEMINI_API_KEY is missing. Please add it in Streamlit → Settings → Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


# ----------------------------------------------------
# 2. Dummy Login Accounts
# ----------------------------------------------------
USER_ACCOUNTS = {
    "user01": "pass01",
    "user02": "pass02",
    "user03": "pass03",
    "user04": "pass04",
    "user05": "pass05",
}


def check_login(uid, pw):
    """Validate the Customer ID + Password."""
    return uid in USER_ACCOUNTS and USER_ACCOUNTS[uid] == pw


# ----------------------------------------------------
# 3. COA Comparison Logic
# ----------------------------------------------------
def generate_comparison(controlled, supplier):
    prompt = f"""
Controlled Specifications:
{controlled}

Supplier Specifications:
{supplier}

Perform a detailed QC-style COA comparison including:
- Mismatches in limits
- Bulk/Tapped density
- Storage conditions
- Particle size
- Critical specification gaps
- Summary + recommendations
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {e}"


# ----------------------------------------------------
# 4. Streamlit Page Layout
# ----------------------------------------------------
st.set_page_config(
    page_title="PharmAiOps – COA Reviewer",
    layout="wide"
)

# Header
st.markdown(
    """
    <h1 style="text-align:center; color:#4A90E2;">
        PharmAiOps – COA Reviewer
    </h1>
    <p style="text-align:center; font-size:18px;">
        AI-powered specification comparison for pharmaceutical QC professionals.
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# 5. Login Section
# ----------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.markdown("### 🔐 Login to access the tool")

    col1, col2 = st.columns(2)

    with col1:
        uid = st.text_input("Customer ID")

    with col2:
        pw = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(uid, pw):
            st.session_state.logged_in = True
            st.success("✔ Login successful!")
            st.experimental_rerun()
        else:
            st.error("❌ Invalid ID or password. Try again.")

    st.stop()


# ----------------------------------------------------
# 6. Main Application UI
# ----------------------------------------------------
st.success("✔ You are logged in.")

# Two Text Areas
col1, col2 = st.columns(2)

with col1:
    controlled_specs = st.text_area(
        "Controlled Specifications",
        placeholder="LOD: NMT 2%...\nAssay: 98–102%...\nDissolution: NLT 80%...",
        height=250
    )

with col2:
    supplier_specs = st.text_area(
        "Supplier Specifications",
        placeholder="LOD: NMT 3%...\nAssay: 98–102%...\nDissolution: NLT 70%...",
        height=250
    )

# Analyze Button
st.markdown("---")
if st.button("🔍 Analyze COA"):
    if not controlled_specs.strip() or not supplier_specs.strip():
        st.error("Please enter both Controlled and Supplier specifications.")
    else:
        with st.spinner("Analyzing with Gemini..."):
            result = generate_comparison(controlled_specs, supplier_specs)
        st.markdown("### 📄 COA Review Result")
        st.markdown(result)

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:13px; color:gray;">
        © 2025 PharmAiOps | AI Tools for Pharmaceutical QC Automation
    </p>
    """,
    unsafe_allow_html=True
)

