import os
import io
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
from pypdf import PdfReader

from openai import OpenAI

# ============================================================
# 0. CONFIG
# ============================================================

APP_TITLE = "AI-Powered Pharma COA Reviewer"
DEVELOPER_NAME = "Developed by Mubashir Hussain"

# Hard coded users with hashed passwords (simple demo, not enterprise auth)
# Generate hashes using hash_password("your_password") and paste here.
VALID_USERS = {
    "mubashir": "put_sha256_hash_here",
    "qc_user": "put_another_hash_here",
}

CRITICAL_PARAMETERS = [
    "assay",
    "identification",
    "identity",
    "residual solvent",
    "residual solvents",
    "microbial limit",
    "microbial limits",
    "microbiology",
    "impurities",
    "related substances",
]

RISK_WEIGHTS = {
    "critical": 3,
    "major": 2,
    "minor": 1,
}

# Material specific rules (you can easily expand this dict)
SPECIAL_MATERIAL_RULES = [
    {
        "pattern": "bromelain",
        "rules": [
            "Verify source of enzyme (e.g. plant origin).",
            "Check enzyme activity as per specification.",
            "Ensure microbial contamination limits are defined and compliant.",
        ],
    },
    {
        "pattern": "extract",
        "rules": [
            "Herbal extract: Verify identity using HPTLC or equivalent assay.",
            "Check assay limits for marker compounds if specified.",
        ],
    },
    {
        "pattern": "potent",
        "rules": [
            "Potent API: Confirm safety limits and OEL considerations.",
            "Verify references to protective handling in COA or SOP.",
        ],
    },
]

# Special API wide rules
API_REQUIRED_TESTS = [
    "residual solvent",
]

EXCIPIENT_FOCUS = [
    "purity",
    "microbial limits",
    "storage condition",
]


# ============================================================
# 1. SECURITY HELPERS
# ============================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def check_credentials(username: str, password: str) -> bool:
    hashed = hash_password(password)
    stored_hash = VALID_USERS.get(username)
    return stored_hash is not None and hashed == stored_hash


def get_openai_client() -> Optional[OpenAI]:
    # Try Streamlit secrets first, then environment
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


# ============================================================
# 2. DATA MODELS
# ============================================================

@dataclass
class ParameterSpec:
    name: str
    spec: str
    source: str  # "monograph" or "controlled_spec"
    criticality: str = "major"  # critical / major / minor


@dataclass
class ParameterResult:
    name: str
    spec: str
    source: str
    coa_value: Optional[str]
    status: str  # "gap", "ok", "not_evaluable"
    comment: str
    risk_level: Optional[str] = None
    risk_score: int = 0


# ============================================================
# 3. PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_chunks = []
    for page in reader.pages:
        try:
            text_chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text_chunks)


# Very simple heuristics; you can improve based on your COA template
def guess_material_and_vendor(coa_text: str) -> Tuple[str, str]:
    material = "Unknown material"
    vendor = "Unknown vendor"

    lines = [l.strip() for l in coa_text.splitlines() if l.strip()]
    if lines:
        material = lines[0]
    for line in lines[:10]:
        lower = line.lower()
        if "manufacturer" in lower or "vendor" in lower or "supplier" in lower:
            vendor = line.split(":", 1)[-1].strip()
            break
    return material, vendor


# ============================================================
# 4. MATERIAL TYPE IDENTIFICATION (API vs EXCIPIENT)
# ============================================================

def identify_material_type_llm(material_name: str, client: Optional[OpenAI]) -> str:
    """
    Simple helper that asks the LLM to classify material as API or excipient.
    In production you can also add a web search layer or internal DB lookup.
    """
    if client is None:
        # Fallback heuristic if API key is not available
        lower = material_name.lower()
        if "sodium" in lower or "chloride" in lower or "powder" in lower:
            return "excipient"
        return "api"

    prompt = (
        f"You are a pharmaceutical expert. "
        f"Classify the material name below strictly as 'API' or 'Excipient'. "
        f"Material: {material_name}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You classify pharma materials for COA review."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content.lower()
        if "excipient" in content:
            return "excipient"
        return "api"
    except Exception:
        # If LLM fails, use simple heuristic
        return "api"


# ============================================================
# 5. SPEC PARSING (MONOGRAPH / CONTROLLED SPEC)
# ============================================================

def parse_specs(text: str, source: str) -> List[ParameterSpec]:
    """
    Very simple parser:
    - Takes lines that contain ':' as parameter: spec
    - You can later replace this with a more robust parser or LLM.
    """
    specs: List[ParameterSpec] = []
    if not text:
        return specs

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        name_part, spec_part = line.split(":", 1)
        name = name_part.strip()
        spec = spec_part.strip()
        if not name or not spec:
            continue

        crit = "major"
        lower_name = name.lower()
        if any(cp in lower_name for cp in CRITICAL_PARAMETERS):
            crit = "critical"

        specs.append(ParameterSpec(name=name, spec=spec, source=source, criticality=crit))
    return specs


# ============================================================
# 6. COA PARSING
# ============================================================

def parse_coa_parameters(text: str) -> Dict[str, str]:
    """
    Simple parameter parsing from COA:
    - Lines that look like "Parameter ...: value"
    - Also capture lines with pattern "Parameter  value" (two columns).
    """
    params: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                params[key.lower()] = value
        else:
            # Try space separated style: first token is name, rest is value
            parts = line.split()
            if len(parts) > 2:
                key = parts[0].strip().lower()
                value = " ".join(parts[1:]).strip()
                if key and value and key not in params:
                    params[key] = value
    return params


def find_coa_value_for_param(param_name: str, coa_params: Dict[str, str]) -> Optional[str]:
    target = param_name.lower()
    # Direct match
    if target in coa_params:
        return coa_params[target]

    # Fuzzy contains match
    for key in coa_params:
        if target in key or key in target:
            return coa_params[key]
    return None


# ============================================================
# 7. RISK ENGINE
# ============================================================

def assess_parameter(param: ParameterSpec, coa_params: Dict[str, str]) -> ParameterResult:
    value = find_coa_value_for_param(param.name, coa_params)

    if value is None:
        status = "gap"
        comment = "Not reported in COA."
    else:
        # For now we do not try to fully parse numeric specifications.
        status = "ok"
        comment = "Reported in COA (detailed numeric check not implemented)."

    risk_level = None
    risk_score = 0
    if status == "gap":
        if param.criticality == "critical":
            risk_level = "critical"
            risk_score = RISK_WEIGHTS["critical"]
        else:
            risk_level = "major"
            risk_score = RISK_WEIGHTS["major"]

    return ParameterResult(
        name=param.name,
        spec=param.spec,
        source=param.source,
        coa_value=value,
        status=status,
        comment=comment,
        risk_level=risk_level,
        risk_score=risk_score,
    )


def generic_risk_checks(
    material_name: str,
    material_type: str,
    coa_text: str,
    coa_params: Dict[str, str],
) -> List[str]:
    notes = []
    lower_text = coa_text.lower()

    # API wide rules
    if material_type == "api":
        for required in API_REQUIRED_TESTS:
            if required not in lower_text:
                notes.append(f"API rule: Residual solvent testing not clearly mentioned for '{material_name}'.")
    # Excipient focus
    if material_type == "excipient":
        if "store" not in lower_text and "storage" not in lower_text:
            notes.append("Excipient rule: Storage condition not defined in COA.")

    # Storage condition missing general
    if "store" not in lower_text and "storage" not in lower_text:
        notes.append("Storage condition missing: Risk of stability or microbial growth.")

    # Special materials
    mlower = material_name.lower()
    for rule_block in SPECIAL_MATERIAL_RULES:
        if rule_block["pattern"] in mlower:
            notes.extend(rule_block["rules"])

    return notes


def calculate_total_risk(results: List[ParameterResult], risk_notes: List[str]) -> int:
    score = sum(r.risk_score for r in results if r.status == "gap")
    # Add small weight for each textual risk note
    score += len(risk_notes)
    return score


# ============================================================
# 8. MAIN REVIEW LOGIC
# ============================================================

def run_coa_review(
    spec_text: str,
    monograph_text: str,
    coa_text: str,
    material_type: str,
    material_name: str,
) -> Dict:
    # Parse all sources
    spec_params = parse_specs(spec_text, source="controlled_spec") if spec_text else []
    mono_params = parse_specs(monograph_text, source="monograph") if monograph_text else []

    coa_params = parse_coa_parameters(coa_text)

    all_specs: List[ParameterSpec] = []
    if mono_params:
        all_specs.extend(mono_params)
    if spec_params:
        all_specs.extend(spec_params)

    results: List[ParameterResult] = []

    if all_specs:
        # Normal comparison flow
        for i, param in enumerate(all_specs, start=1):
            res = assess_parameter(param, coa_params)
            results.append(res)
    else:
        # No specs or monograph available: full risk based only.
        results = []

    # Generic risk checks not tied to individual parameter rows
    risk_notes = generic_risk_checks(material_name, material_type, coa_text, coa_params)

    # Convert to compliance table DF (only gaps as per requirement)
    gap_rows = []
    serial = 1
    for r in results:
        if r.status != "gap":
            continue
        risk_flag = r.risk_level or "major"
        gap_rows.append({
            "S. No.": serial,
            "Parameter": r.name,
            "Specification": r.spec,
            "Source": r.source,
            "Gap / Comment": r.comment,
            "Risk Level": risk_flag,
        })
        serial += 1

    compliance_df = pd.DataFrame(gap_rows) if gap_rows else pd.DataFrame(
        columns=["S. No.", "Parameter", "Specification", "Source", "Gap / Comment", "Risk Level"]
    )

    total_risk_score = calculate_total_risk(results, risk_notes)

    return {
        "compliance_df": compliance_df,
        "risk_notes": risk_notes,
        "total_risk_score": total_risk_score,
        "raw_results": results,
    }


# ============================================================
# 9. EMAIL GENERATION / REPORTING
# ============================================================

def format_risk_notes_bullets(risk_notes: List[str]) -> str:
    if not risk_notes:
        return "None."
    return "\n".join([f"- {n}" for n in risk_notes])


def generate_email(
    material_name: str,
    vendor_name: str,
    compliance_df: pd.DataFrame,
    risk_notes: List[str],
) -> Tuple[str, str]:
    subject = f"COA review for {material_name} from {vendor_name} – Gaps and risk assessment"

    gaps_text_lines = []
    if not compliance_df.empty:
        for _, row in compliance_df.iterrows():
            gaps_text_lines.append(
                f"{int(row['S. No.'])}. {row['Parameter']} "
                f"(Spec: {row['Specification']}) – Gap: {row['Gap / Comment']} "
                f"[Risk: {row['Risk Level']}]"
            )
    else:
        gaps_text_lines.append("No gaps identified based on available specs/monograph.")

    gaps_text = "\n".join(gaps_text_lines)
    risk_bullets = format_risk_notes_bullets(risk_notes)

    body = f"""
Dear Sir / Madam,

Subject: COA review for {material_name} from {vendor_name}

We have reviewed the submitted Certificate of Analysis (COA) against the available pharmacopoeial monograph and/or controlled specification.

Summary of Gaps:
{gaps_text}

Risk Considerations:
{risk_bullets}

Recommended Corrective Actions:
- Please update the COA to include all missing critical parameters (e.g. assay, identification, residual solvents, microbial limits) as applicable.
- Where storage conditions are not clearly defined, add a clear storage statement to support product stability and GMP compliance.
- For any high risk gaps, perform additional testing internally or through an external laboratory before batch release.

Once the updated COA or additional test reports are available, we can close this review.

Regards,
Quality Control / Quality Assurance
"""

    return subject.strip(), body.strip()


def build_text_report(
    material_name: str,
    vendor_name: str,
    compliance_df: pd.DataFrame,
    risk_notes: List[str],
    total_risk_score: int,
) -> str:
    # Simple text file for SCM
    buffer = []
    buffer.append(f"Material: {material_name}")
    buffer.append(f"Vendor: {vendor_name}")
    buffer.append(f"Total risk score: {total_risk_score}")
    buffer.append("")
    buffer.append("Compliance gaps (monograph/spec based):")
    buffer.append("-" * 60)

    if compliance_df.empty:
        buffer.append("No gaps identified.")
    else:
        for _, row in compliance_df.iterrows():
            buffer.append(
                f"{int(row['S. No.'])}. {row['Parameter']} "
                f"(Spec: {row['Specification']}, Source: {row['Source']})"
            )
            buffer.append(f"   Gap: {row['Gap / Comment']} [Risk: {row['Risk Level']}]")

    buffer.append("")
    buffer.append("Risk notes / special considerations:")
    buffer.append("-" * 60)
    if not risk_notes:
        buffer.append("None.")
    else:
        for note in risk_notes:
            buffer.append(f"- {note}")

    return "\n".join(buffer)


# ============================================================
# 10. STREAMLIT UI
# ============================================================

def setup_page():
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Simple CSS theming
    st.markdown(
        """
        <style>
        .app-header {
            background-color: #1f4e79;
            color: white;
            padding: 0.6rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .app-footer {
            margin-top: 2rem;
            font-size: 0.8rem;
            color: #666666;
            text-align: center;
        }
        .risk-critical {
            color: #ffffff;
            background-color: #b00020;
            padding: 0.2rem 0.4rem;
            border-radius: 0.3rem;
        }
        .risk-major {
            color: #ffffff;
            background-color: #f57c00;
            padding: 0.2rem 0.4rem;
            border-radius: 0.3rem;
        }
        .risk-minor {
            color: #ffffff;
            background-color: #388e3c;
            padding: 0.2rem 0.4rem;
            border-radius: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='app-header'><h3>{APP_TITLE}</h3></div>", unsafe_allow_html=True)


def login_section():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    st.subheader("Login")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if check_credentials(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.success("Login successful.")
                return True
            else:
                st.error("Invalid username or password.")
                return False

    return False


def main_app():
    client = get_openai_client()

    st.subheader("Input files")

    col1, col2, col3 = st.columns(3)

    with col1:
        spec_file = st.file_uploader(
            "Controlled specification (optional)",
            type=["pdf"],
            help="Upload internal controlled specification. If not available, the app will use monograph and risk based rules.",
        )
    with col2:
        monograph_file = st.file_uploader(
            "Pharmacopoeial monograph (optional)",
            type=["pdf"],
            help="Upload relevant pharmacopoeial monograph. If not available, the app will use controlled specification and risk based rules.",
        )
    with col3:
        coa_file = st.file_uploader(
            "Vendor COA (required)",
            type=["pdf"],
            help="Vendor Certificate of Analysis. This is mandatory.",
        )

    st.markdown("#### Options")

    material_name_override = st.text_input(
        "Material name (optional – will override COA derived name)",
        value="",
    )
    vendor_name_override = st.text_input(
        "Vendor name (optional – will override COA derived name)",
        value="",
    )

    col_type1, col_type2 = st.columns(2)
    with col_type1:
        material_type_override = st.selectbox(
            "Material type",
            options=["Auto detect (LLM)", "API", "Excipient"],
            help="You can force API/Excipient type or let the model classify based on material name.",
        )
    with col_type2:
        st.caption("If neither monograph nor controlled spec is uploaded, app will run full risk based COA review.")

    process_btn = st.button("Run COA Review")

    if not process_btn:
        return

    if coa_file is None:
        st.error("Vendor COA is required.")
        return

    with st.spinner("Reviewing COA and applying risk based checks..."):
        # Extract text
        spec_text = extract_text_from_pdf(spec_file) if spec_file else ""
        monograph_text = extract_text_from_pdf(monograph_file) if monograph_file else ""
        coa_text = extract_text_from_pdf(coa_file)

        # Material and vendor
        guessed_material, guessed_vendor = guess_material_and_vendor(coa_text)

        material_name = material_name_override.strip() or guessed_material
        vendor_name = vendor_name_override.strip() or guessed_vendor

        # Material type
        if material_type_override == "API":
            material_type = "api"
        elif material_type_override == "Excipient":
            material_type = "excipient"
        else:
            material_type = identify_material_type_llm(material_name, client)

        results = run_coa_review(
            spec_text=spec_text,
            monograph_text=monograph_text,
            coa_text=coa_text,
            material_type=material_type,
            material_name=material_name,
        )

        compliance_df = results["compliance_df"]
        risk_notes = results["risk_notes"]
        total_risk_score = results["total_risk_score"]

        email_subject, email_body = generate_email(
            material_name=material_name,
            vendor_name=vendor_name,
            compliance_df=compliance_df,
            risk_notes=risk_notes,
        )

        text_report = build_text_report(
            material_name=material_name,
            vendor_name=vendor_name,
            compliance_df=compliance_df,
            risk_notes=risk_notes,
            total_risk_score=total_risk_score,
        )

        # Store in session history
        if "history" not in st.session_state:
            st.session_state["history"] = []
        st.session_state["history"].append(
            {
                "material": material_name,
                "vendor": vendor_name,
                "risk_score": total_risk_score,
                "compliance_df": compliance_df,
                "risk_notes": risk_notes,
            }
        )

    # ========================================================
    # Output sections
    # ========================================================

    st.markdown(f"### Result summary for **{material_name}** ({material_type.upper()})")

    # Risk badge
    risk_label = "Low"
    if total_risk_score >= 10:
        risk_label = "High"
        css_class = "risk-critical"
    elif total_risk_score >= 5:
        risk_label = "Medium"
        css_class = "risk-major"
    else:
        css_class = "risk-minor"

    st.markdown(
        f"Total risk score: <span class='{css_class}'>{total_risk_score} ({risk_label})</span>",
        unsafe_allow_html=True,
    )

    # Compliance table
    with st.expander("Compliance table (monograph / controlled spec based)", expanded=True):
        if compliance_df.empty:
            st.success("No gaps identified based on parsed specifications.")
        else:
            def color_rows(row):
                if row["Risk Level"] == "critical":
                    return ["background-color: #ffebee"] * len(row)
                elif row["Risk Level"] == "major":
                    return ["background-color: #fff3e0"] * len(row)
                else:
                    return ["background-color: #e8f5e9"] * len(row)

            styled = compliance_df.style.apply(color_rows, axis=1)
            st.dataframe(styled, use_container_width=True)

    # Risk notes
    with st.expander("Risk analysis notes", expanded=True):
        if not risk_notes:
            st.info("No additional risk notes identified.")
        else:
            for note in risk_notes:
                st.markdown(f"- {note}")

    # Email preview
    with st.expander("Email preview", expanded=False):
        st.text_area("Email subject", value=email_subject, height=40)
        st.text_area("Email body", value=email_body, height=300)

    # Downloadable report
    with st.expander("Downloadable report for SCM", expanded=False):
        st.download_button(
            label="Download text report",
            data=text_report,
            file_name=f"COA_review_{material_name.replace(' ', '_')}.txt",
            mime="text/plain",
        )

    # Session history
    with st.expander("Session history (current login)", expanded=False):
        history = st.session_state.get("history", [])
        if not history:
            st.write("No previous reviews in this session.")
        else:
            hist_rows = []
            for idx, item in enumerate(history, start=1):
                hist_rows.append(
                    {
                        "S. No.": idx,
                        "Material": item["material"],
                        "Vendor": item["vendor"],
                        "Risk score": item["risk_score"],
                    }
                )
            hist_df = pd.DataFrame(hist_rows)
            st.dataframe(hist_df, use_container_width=True)

    # Footer
    st.markdown(
        f"<div class='app-footer'>{DEVELOPER_NAME} | This tool supports "
        f"risk based COA review using controlled specs, monographs and ICH style rules.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# 11. MAIN ENTRY
# ============================================================

def main():
    setup_page()
    logged_in = login_section()
    if not logged_in:
        st.markdown(
            f"<div class='app-footer'>{DEVELOPER_NAME}</div>",
            unsafe_allow_html=True,
        )
        return

    main_app()


if __name__ == "__main__":
    main()
