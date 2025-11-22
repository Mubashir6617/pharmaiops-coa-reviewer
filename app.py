import os
import google.generativeai as genai
import gradio as gr


# ============================
# 1. Gemini API Configuration
# ============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your HF Secrets.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")


# ============================
# 2. Dummy User Accounts
# ============================
USER_ACCOUNTS = {
    "user01": "pass01",
    "user02": "pass02",
    "user03": "pass03",
    "user04": "pass04",
    "user05": "pass05"
}


def authenticate(user_id, password):
    """Check if ID & password match."""
    if user_id in USER_ACCOUNTS and USER_ACCOUNTS[user_id] == password:
        return True, "Login successful!"
    return False, "Invalid ID or Password."


# ============================
# 3. COA Comparison Function
# ============================
def generate_intelligent_comparison(controlled_text: str, supplier_text: str) -> str:
    prompt = f"""
Controlled Specifications:
{controlled_text}

Supplier Specifications:
{supplier_text}

Perform a detailed QC-style comparison between these specifications.
Highlight mismatches, storage, particle size, bulk/tapped density etc.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"


def compare_specs(controlled_text, supplier_text):
    if not controlled_text.strip():
        return "Please enter controlled specifications."
    if not supplier_text.strip():
        return "Please enter supplier specifications."
    return generate_intelligent_comparison(controlled_text, supplier_text)


# ============================
# 4. Gradio UI (With Login)
# ============================
with gr.Blocks() as demo:

    gr.Markdown("# 🔐 PharmAiOps – COA Reviewer")
    gr.Markdown("### Login with Customer ID & Password")

    # LOGIN UI
    user_id = gr.Textbox(label="Customer ID", placeholder="Enter your ID")
    password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
    login_btn = gr.Button("Login")
    login_status = gr.Markdown()

    # MAIN TOOL UI (hidden by default)
    controlled_box = gr.Textbox(label="Controlled Specs", lines=10, visible=False)
    supplier_box = gr.Textbox(label="Supplier Specs", lines=10, visible=False)
    analyze_btn = gr.Button("Analyze COA", visible=False)
    output = gr.Markdown(visible=False)


    # LOGIN LOGIC
    def login_action(uid, pw):
        ok, msg = authenticate(uid, pw)
        if ok:
            return (
                gr.update(visible=False),   # Hide ID field
                gr.update(visible=False),   # Hide PW field
                gr.update(value="### ✔️ Login Successful! Welcome.", visible=True),
                gr.update(visible=True),    # Controlled box visible
                gr.update(visible=True),    # Supplier box visible
                gr.update(visible=True),    # Analyze button
                gr.update(visible=True),    # Output area
            )
        else:
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(value="❌ Invalid login. Try again.", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    login_btn.click(
        login_action,
        inputs=[user_id, password],
        outputs=[user_id, password, login_status, controlled_box, supplier_box, analyze_btn, output]
    )


    # ANALYSIS LOGIC
    analyze_btn.click(compare_specs, inputs=[controlled_box, supplier_box], outputs=output)


if __name__ == "__main__":
    demo.launch()
