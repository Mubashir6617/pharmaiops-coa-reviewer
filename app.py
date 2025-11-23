import io
import os
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pdfplumber
from PIL import Image

# Optional: OCR
try:
    import pytesseract
except ImportError:
    pytesseract = None

# Optional: .env support for local dev (not required in production)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# PDF generation for SCM report
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile

# ---------------------------
# Streamlit Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AI-Powered Pharma COA Reviewer",
    page_icon="🧪",
    layout="wide",
)

# Footer
st.markdown(
    """
<hr>
<div style="text-align: center; font-size: 14px; color: gray;">
    Developed by <b>Mubashir Hussain</b>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Configuration / Constants
# ---------------------------
# Hard-coded users (password hash is for "123456")
USERS: Dict[str, str] = {
    "mubashir": "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92",
    "user1": "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92",
    "user2": "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92",
    "user3": "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92",
    "user4": "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92",
    "user5": "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92",
}

MODEL_OPTIONS = [
    "gpt-5.1",
    "gpt-5 mini",
    "gpt-5 nano",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]


# ---------------------------
# Utility: Auth
# ---------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_user(username: str, password: str) -> bool:
    return USERS.get(username) == hash_password(password)


def _authenticate() -> bool:
    if st.session_state.get("authenticated"):
        return True

    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Sign in"):
        if verify_user(username, password):
            st.session_state["authenticated"] = True
            st.sidebar.success("Authenticated")
            return True
        st.sidebar.error("Invalid credentials")
    return False


# ---------------------------
# Utility: OpenAI Client
# ---------------------------
def get_openai_client() -> Optional[Any]:
    """
    Preferred order:
    1. st.secrets["OPENAI_API_KEY"]
    2. os.environ["OPENAI_API_KEY"]

    If missing or SDK not installed, show a clear error.
    """
    if OpenAI is None:
        st.error("OpenAI Python SDK is not installed. Add `openai` to requirements.txt.")
        return None

    api_key: Optional[str] = None

    # Try Streamlit secrets
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None

    # Fallback to environment variable (for local dev)
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "OpenAI API key not found. "
            "Set `OPENAI_API_KEY` in `.streamlit/secrets.toml` or as an environment variable."
        )
        return None

    return OpenAI(api_key=api_key)


# ---------------------------
# PDF Text Extraction
# ---------------------------
def _load_page_images(uploaded_file: io.BytesIO) -> List[Image.Image]:
    images: List[Image.Image] = []
    uploaded_file.seek(0)
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            pil_image = page.to_image(resolution=300).original
            images.append(pil_image)
    return images


def extract_pdf_text(uploaded_file: io.BytesIO) -> str:
    """
    1. Try pdfplumber text extraction.
    2. If empty, fallback to OCR (if pytesseract is available).
    """
    uploaded_file.seek(0)
    text_fragments: List[str] = []

    # Step 1: pdfplumber
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text_fragments.append(page.extract_text() or "")

    text = "\n".join(text_fragments).strip()
    if text:
        return text

    # Step 2: OCR fallback
    uploaded_file.seek(0)
    ocr_fragments: List[str] = []
    if pytesseract is None:
        return "[ERROR] No text found via pdfplumber and Tesseract OCR is not installed."

    for page_image in _load_page_images(uploaded_file):
        ocr_fragments.append(pytesseract.image_to_string(page_image))

    return "\n".join(ocr_fragments)


# ---------------------------
# Markdown Table Utilities
# ---------------------------
def parse_markdown_table(table_text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for line in table_text.splitlines():
        if "|" not in line:
            continue
        if set(line.strip()) in [{"-"}, {"|"}, {"-", "|"}]:
            continue

        cells = [cell.strip() for cell in line.split("|") if cell.strip()]
        if len(cells) < 2:
            continue

        if cells[0].lower() in {"parameter", "s. no.", "s.no"}:
            continue

        rows.append({"parameter": cells[0], "spec": cells[1]})

    return rows


def standard_table(rows: List[Tuple[str, str]]) -> str:
    header = ["Parameter", "Specification"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|---|",
    ]
    for parameter, spec in rows:
        lines.append(f"| {parameter.strip()} | {spec.strip()} |")
    return "\n".join(lines)


# ---------------------------
# LLM Extraction
# ---------------------------
def run_extractor(
    client: Any, model_name: str, kind: str, text: str
) -> str:
    prompts = {
        "usp": "Extract only test parameters and acceptance criteria from the USP monograph.",
        "bp": "Extract only test parameters and acceptance criteria from the BP monograph.",
        "controlled": "Extract only test parameters and acceptance criteria from the controlled specification.",
        "vendor": "Extract only test parameters and acceptance criteria from the vendor COA.",
    }

    prompt_header = prompts.get(kind, "Extract parameters and acceptance criteria.")

    instructions = (
        f"{prompt_header} "
        "You are a pharmaceutical monograph and COA interpretation expert operating at GPT-5.1 reasoning level. "
        "Extract only the true analytical test parameters and their acceptance criteria from the document.\n\n"
    
        "Output Requirements:\n"
        "- Return a clean, strict markdown table with columns: `Parameter` and `Specification`.\n"
        "- Ensure each parameter corresponds to a real QC test (e.g., Description, Identification, Assay, Impurities, pH, LOD, Residue on Ignition, Specific Optical Rotation, etc.).\n"
        "- For identification tests, summarize the acceptance criteria (e.g., 'Conforms', 'IR matches reference standard').\n"
        "- For assays and impurities, include the exact numeric ranges or limits.\n"
        "- For limit tests (e.g., Arsenic, Heavy metals), extract the exact NMT/NLT value.\n"
        "- Shorten long textual acceptance criteria into concise, accurate phrases.\n\n"
    
        "Strict Exclusions:\n"
        "- Do NOT include system suitability criteria (RSD, tailing, theoretical plates).\n"
        "- Do NOT include chromatography conditions, mobile phases, detector parameters.\n"
        "- Do NOT include apparatus details or sample preparation steps.\n"
        "- Do NOT fabricate or infer any missing value.\n\n"
    
        "General Rules:\n"
        "- Keep each `Parameter` name clean and standardised.\n"
        "- Keep each `Specification` short, factual, and aligned with the document.\n"
        "- If a parameter has multiple criteria, merge them into one concise specification.\n"
    )
  

    truncated_text = text[:6000] if text else ""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You extract pharmaceutical quality parameters into a two-column markdown table.",
            },
            {
                "role": "user",
                "content": f"{instructions}\n\nDocument text:\n{truncated_text}",
            },
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


# ---------------------------
# Comparison Utilities
# ---------------------------
def _normalize_param(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower()).strip()


def _parse_numeric_range(spec: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    if not spec:
        return None
    numbers = [float(match) for match in re.findall(r"-?\d+\.?\d*", spec)]
    if not numbers:
        return None

    spec_lower = spec.lower()

    if "nmt" in spec_lower or "not more than" in spec_lower:
        return (None, numbers[-1])
    if "nlt" in spec_lower or "not less than" in spec_lower:
        return (numbers[0], None)
    if len(numbers) >= 2:
        return (min(numbers), max(numbers))
    return (numbers[0], numbers[0])


def evaluate_status(
    vendor_spec: Optional[str], ref_spec: Optional[str]
) -> Tuple[str, str]:
    if vendor_spec is None:
        return ("Critical gap – parameter missing in COA", "🟥")
    if ref_spec is None:
        return ("OK – reviewed against vendor values only", "🟩")

    vendor_range = _parse_numeric_range(vendor_spec)
    ref_range = _parse_numeric_range(ref_spec)

    if vendor_range and ref_range:
        v_min, v_max = vendor_range
        r_min, r_max = ref_range

        if r_min is not None and v_min is not None and v_min < r_min:
            return ("Major gap – vendor limit below reference minimum", "🟧")
        if r_max is not None and v_max is not None and v_max > r_max:
            return ("Major gap – vendor limit above reference maximum", "🟧")

    return ("OK – parameter aligned with reference", "🟩")


def compare_tables(
    vendor_rows: List[Dict[str, str]],
    reference_rows: List[Dict[str, str]],
    reference_name: str,
) -> List[Dict[str, str]]:
    reference_map = {_normalize_param(r["parameter"]): r["spec"] for r in reference_rows}
    vendor_map = {_normalize_param(r["parameter"]): r["spec"] for r in vendor_rows}

    all_params = set(reference_map.keys()) | set(vendor_map.keys())
    results: List[Dict[str, str]] = []

    for param_key in sorted(all_params):
        vendor_spec = vendor_map.get(param_key)
        ref_spec = reference_map.get(param_key)

        display_name = next(
            (
                r["parameter"]
                for r in vendor_rows + reference_rows
                if _normalize_param(r["parameter"]) == param_key
            ),
            param_key.title(),
        )

        status, color = evaluate_status(vendor_spec, ref_spec)
        spec_display = vendor_spec or ref_spec or "Not reported"
        results.append(
            {
                "Parameter": display_name,
                "Specification": spec_display,
                "Status": f"{color} {status} ({reference_name})",
            }
        )

    return results


def build_gap_list(comparison_rows: List[Dict[str, str]]) -> List[str]:
    gaps: List[str] = []
    for row in comparison_rows:
        if row["Status"].startswith("🟩"):
            continue
        parameter = row["Parameter"]
        status = row["Status"].split(" ", 1)[1]
        gaps.append(f"{parameter}: {status}")
    return gaps


# ---------------------------
# Risk Notes & Email Text
# ---------------------------
def generate_risk_notes(
    client: Any, model_name: str, gaps: List[str]
) -> str:
    if not gaps:
        return "No critical risks identified. Routine monitoring and periodic vendor requalification are recommended."

    prompt = (
        "You are an advanced pharmaceutical QA/QC and GMP compliance specialist with deep expertise in:\n"
        "- ICH Q7, Q8, Q9, Q10\n"
        "- WHO TRS, FDA 21 CFR Part 211, and PIC/S guidelines\n\n"
        "Analyze the COA gaps provided below and generate **clear, concise, risk-based notes**.\n"
        "Your analysis must reflect GPT-5.1 level reasoning: accurate, structured, and context-aware.\n\n"
        "Include:\n"
        "1. **Quality & safety impact:** How the gap could affect assay, impurities, stability, identification, or patient safety.\n"
        "2. **Regulatory risk:** Compliance implications relative to pharmacopeial limits or GMP expectations.\n"
        "3. **Recommended actions:** Practical, audit-ready steps such as repeat testing, requesting updated COA, investigation, CAPA, or vendor requalification.\n\n"
        "Output style:\n"
        "- Write in short, crisp bullets (without heading and in short, simple human english).\n"
        "- Do not repeat the gap list; interpret it.\n"
        "- Keep tone professional, objective, and aligned with pharmaceutical QA language.\n"
       )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n".join(gaps)},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


def generate_email(material: str, manufacturer: str, gaps: List[str]) -> str:
    gap_lines = "\n".join(f"- {gap}" for gap in gaps) or "- No major gaps identified."

    return (
        f"Subject: COA Review for {material or 'Material'} – Gaps and Risk Summary\n\n"
        f"Dear SCM,\n\n"
        f"Kindly find attached the reviewed COA of {material or 'the material'} "
        f"from {manufacturer or 'the manufacturer'}.\n\n"
        f"Below are the key gaps and observations:\n{gap_lines}\n\n"
        f"Regards,\n"
        f"AI-Powered COA Reviewer\n"
    )


# ---------------------------
# Report Generation (Markdown + PDF)
# ---------------------------
def generate_report_markdown(
    material: str,
    manufacturer: str,
    vendor_table: str,
    reference_name: str,
    reference_table: str,
    comparison_rows: List[Dict[str, str]],
    gaps: List[str],
    risk_notes: str,
) -> str:
    lines: List[str] = []

    lines.append("# AI-Powered Pharma COA Reviewer Report")
    lines.append("")
    lines.append(f"**Material:** {material or 'N/A'}")
    lines.append(f"**Manufacturer:** {manufacturer or 'N/A'}")
    lines.append("")
    lines.append("## Vendor COA Extraction")
    lines.append("")
    lines.append(vendor_table or "_No data extracted_")
    lines.append("")
    lines.append(f"## Reference Extraction ({reference_name})")
    lines.append("")
    lines.append(reference_table or "_No reference document provided_")
    lines.append("")
    lines.append("## Compliance Review")
    lines.append("")
    if comparison_rows:
        for idx, row in enumerate(comparison_rows, start=1):
            lines.append(
                f"{idx}. **{row['Parameter']}** | {row['Specification']} | {row['Status']}"
            )
    else:
        lines.append("_No comparison performed_")
    lines.append("")
    lines.append("## Gaps")
    lines.append("")
    if gaps:
        for gap in gaps:
            lines.append(f"- {gap}")
    else:
        lines.append("No significant gaps identified.")
    lines.append("")
    lines.append("## Risk Analysis")
    lines.append("")
    lines.append(risk_notes or "_No risk notes generated_")

    return "\n".join(lines)


def generate_pdf_from_text(report_text: str) -> bytes:
    """
    Simple text-to-PDF: write each line, paginate when needed.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        c = canvas.Canvas(tmp.name, pagesize=A4)
        width, height = A4
        left_margin = 40
        top_margin = height - 40
        line_height = 14

        y = top_margin

        for line in report_text.split("\n"):
            # Basic wrapping: if line is very long, split it
            while len(line) > 110:
                part = line[:110]
                line = line[110:]
                c.drawString(left_margin, y, part)
                y -= line_height
                if y < 40:
                    c.showPage()
                    y = top_margin
            c.drawString(left_margin, y, line)
            y -= line_height
            if y < 40:
                c.showPage()
                y = top_margin

        c.save()
        tmp.seek(0)
        return tmp.read()


# ---------------------------
# Reference Selection
# ---------------------------
def select_reference(
    usp_table: str, bp_table: str, spec_table: str
) -> Tuple[str, str]:
    if usp_table:
        return usp_table, "USP"
    if bp_table:
        return bp_table, "BP"
    if spec_table:
        return spec_table, "Controlled Spec"
    # Last resort
    fallback = standard_table(
        [("Self Review", "No pharmacopoeial or controlled spec provided – reviewed against vendor values only.")]
    )
    return fallback, "Vendor self-review"


# ---------------------------
# UI Helpers
# ---------------------------
def _show_status_banner():
    st.markdown(
        """
        ### 🌟 AI-Powered Pharma COA Reviewer
        Developed by **Mubashir Hussain**

        **Purpose:**  
        Review vendor COAs against USP/BP/controlled specifications using OpenAI, 
        generate gap analysis, risk commentary, SCM email text, and a PDF report.
        """
    )


def _display_tables_section(title: str, table_text: str):
    st.subheader(title)
    st.markdown(table_text)


def _render_compliance_table(rows: List[Dict[str, str]]):
    if not rows:
        st.info("No comparison available yet.")
        return
    st.subheader("Compliance Table")
    st.dataframe(rows, use_container_width=True)


# ---------------------------
# Main App
# ---------------------------
def main():
    _show_status_banner()

    # Authentication first
    if not _authenticate():
        st.stop()

    # Sidebar: model and uploads
    st.sidebar.header("Model & Documents")

    model_choice = st.sidebar.selectbox(
        "Select OpenAI model",
        MODEL_OPTIONS,
        index=0,
    )

    vendor_file = st.sidebar.file_uploader("Vendor COA PDF (required)", type=["pdf"])
    usp_file = st.sidebar.file_uploader("USP Monograph PDF (optional)", type=["pdf"])
    bp_file = st.sidebar.file_uploader("BP Monograph PDF (optional)", type=["pdf"])
    spec_file = st.sidebar.file_uploader("Controlled Specification PDF (optional)", type=["pdf"])

    material = st.text_input("Material Name")
    manufacturer = st.text_input("Manufacturer Name")

    if st.button("Run COA Review", type="primary"):
        if not vendor_file:
            st.error("Vendor COA is required to start the review.")
            st.stop()

        client = get_openai_client()
        if not client:
            st.stop()

        with st.spinner("Extracting text from PDFs..."):
            vendor_text = extract_pdf_text(vendor_file)
            usp_text = extract_pdf_text(usp_file) if usp_file else ""
            bp_text = extract_pdf_text(bp_file) if bp_file else ""
            spec_text = extract_pdf_text(spec_file) if spec_file else ""

        with st.spinner("Running LLM extraction..."):
            vendor_table = run_extractor(client, model_choice, "vendor", vendor_text)
            usp_table = run_extractor(client, model_choice, "usp", usp_text) if usp_text else ""
            bp_table = run_extractor(client, model_choice, "bp", bp_text) if bp_text else ""
            spec_table = run_extractor(client, model_choice, "controlled", spec_text) if spec_text else ""

        reference_table, reference_name = select_reference(usp_table, bp_table, spec_table)

        vendor_rows = parse_markdown_table(vendor_table)
        reference_rows = parse_markdown_table(reference_table)
        comparison_rows = compare_tables(vendor_rows, reference_rows, reference_name)

        gaps = build_gap_list(comparison_rows)
        risk_notes = generate_risk_notes(client, model_choice, gaps)
        email_text = generate_email(material, manufacturer, gaps)
        report_markdown = generate_report_markdown(
            material,
            manufacturer,
            vendor_table,
            reference_name,
            reference_table,
            comparison_rows,
            gaps,
            risk_notes,
        )

        pdf_bytes = generate_pdf_from_text(report_markdown)

        # Layout
        cols = st.columns(2)
        with cols[0]:
            _display_tables_section("Vendor COA Extracted Table", vendor_table)
            if usp_table:
                _display_tables_section("USP Extracted Table", usp_table)
            if bp_table:
                _display_tables_section("BP Extracted Table", bp_table)

        with cols[1]:
            if spec_table:
                _display_tables_section("Controlled Specification Table", spec_table)
            _render_compliance_table(comparison_rows)
            st.subheader("Risk Analysis Notes")
            st.write(risk_notes)

        st.subheader("Auto-Generated Email to SCM")
        st.code(email_text, language="markdown")

        st.subheader("Download SCM Report")
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="scm_coa_review_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()

