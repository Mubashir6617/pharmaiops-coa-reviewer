import os
import google.generativeai as genai
import gradio as gr


# ----------------------------
# 1. Configure Gemini from ENV
# ----------------------------
# IMPORTANT: Do NOT hard-code your API key.
# On Hugging Face, you will set GEMINI_API_KEY as a secret.
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    # In local dev, you can uncomment and temporarily set it here,
    # but NEVER commit your real key to Git or Hugging Face.
      API_KEY = "YOUR_KEY_HERE"
    raise RuntimeError(
        "GEMINI_API_KEY environment variable not set. "
        "Set it locally or in your Hugging Face Space secrets."
    )

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-pro")


# --------------------------------------------
# 2. Your original logic, refactored as a func
# --------------------------------------------
def generate_intelligent_comparison(controlled_text: str, supplier_text: str) -> str:
    prompt = f"""
In the QC (Quality Control) laboratory of a pharmaceutical company, we perform detailed comparisons of the COAs (Certificates of Analysis) for raw materials used in pharmaceutical manufacturing. The goal is to ensure that raw materials meet the required specifications to guarantee product quality and consistency.

Controlled Specifications:
{controlled_text}

Supplier Specifications:
{supplier_text}

Important Terminology Clarification:
- "LOD" and "Loss on Drying" refer to the same parameter in pharmaceutical analysis and should be considered equivalent.
- Please ensure that any other terms used interchangeably (e.g., bulk density and tapped density) are also recognized as matching if they are contextually equivalent.

Please conduct a thorough comparison of these specifications, focusing on the following key aspects:

1. Mismatch Report: Identify any discrepancies between the controlled and supplier specifications, including both the parameters and their defined limits. Explain why these discrepancies are significant for raw material quality and product performance, including terms such as LOD/ Loss on Drying, bulk density, etc.

2. Bulk and Tapped Density: Check if the supplier's COA includes values for bulk and tapped density. If these values are missing or not aligned, highlight the absence and explain the importance of these parameters for the handling, formulation, and flow properties of the raw material.

3. Storage Conditions: Carefully review and compare the storage conditions provided in both the controlled and supplier specifications. Discuss any mismatches and the potential impact of improper storage conditions on the raw material's stability, potency, and shelf-life.

4. Particle Size Specification: Verify whether the supplier's COA includes a particle size specification. If it is missing, indicate this omission and explain why particle size is a critical parameter for ensuring consistency, dissolution rate, and bioavailability of the raw material in the final product.

5. Overall Summary: Provide a comprehensive summary of the alignment between the controlled and supplier specifications. Offer specific recommendations for improvements or adjustments needed to ensure the supplier's COA aligns with the controlled specifications, ensuring that the raw materials meet the necessary quality standards for pharmaceutical use.

Please focus on delivering a precise and actionable analysis, highlighting critical mismatches, and providing guidance on how to address them to maintain high-quality standards.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error occurred during AI processing: {e}"


# -------------------------------------------------
# 3. Wrapper for Gradio UI (validation + formatting)
# -------------------------------------------------
def compare_specs(controlled_text: str, supplier_text: str) -> str:
    if not controlled_text.strip():
        return "Please enter the controlled specifications."
    if not supplier_text.strip():
        return "Please enter the supplier specifications."

    return generate_intelligent_comparison(controlled_text, supplier_text)


# --------------------------
# 4. Gradio Web Application
# --------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        """
# PharmAiOps – COA Spec Comparator
AI that compares controlled vs supplier COA specifications like a pharmaceutical professional.

Paste your **controlled specs** and **supplier specs** in natural language
(e.g. “Dissolution: NLT 80% (Q), Assay: 98–102%, LOD NMT 2%”).
"""
    )

    with gr.Row():
        controlled_box = gr.Textbox(
            label="Controlled Specifications",
            placeholder="e.g. Dissolution: NLT 80% (Q), Assay: 98–102%, LOD NMT 2%, …",
            lines=12,
        )
        supplier_box = gr.Textbox(
            label="Supplier Specifications",
            placeholder="e.g. Dissolution: NLT 70% (Q), Assay: 98–102%, LOD NMT 3%, …",
            lines=12,
        )

    run_button = gr.Button("Analyze & Compare")
    output_md = gr.Markdown(label="Intelligent Comparison Results")

    run_button.click(
        fn=compare_specs,
        inputs=[controlled_box, supplier_box],
        outputs=output_md,
    )

if __name__ == "__main__":
    demo.launch()
