from __future__ import annotations

import html
from urllib.parse import quote
import time
from pathlib import Path

import gradio as gr
import torch

from dataset import (
    build_visual_inputs,
    make_messages,
)
from model import (
    build_processor,
    load_adapter_for_inference,
)
from predict import parse_prediction
from utils import load_yaml


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "final_model.yaml"
)

ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "final_model"
    / "final_adapter"
)


# ============================================================
# Load final model configuration
# ============================================================

print("\n========================================")
print(" SHM Structural Damage Analysis Demo")
print("========================================")

print("\nLoading configuration...")

config = load_yaml(CONFIG_PATH)

model_cfg = config["model"]
data_cfg = config["data"]

visual_input_mode = str(
    data_cfg.get("visual_input_mode", "single_image")
)
num_local_crops = int(
    data_cfg.get("num_local_crops", 2)
)
crop_fraction = float(
    data_cfg.get("crop_fraction", 0.65)
)
prompt_variant = str(
    data_cfg.get("prompt_variant", "direct_free_text")
)


# ============================================================
# Load processor
# ============================================================

print("Loading processor...")

processor = build_processor(
    model_cfg["model_id"],
    min_pixels=int(model_cfg["min_pixels"]),
    max_pixels=int(model_cfg["max_pixels"]),
)

processor.tokenizer.padding_side = "left"


# ============================================================
# Load FINAL language-adapted LoRA model
# ============================================================

print("Loading final Qwen3.5-4B adapter...")

if not ADAPTER_PATH.exists():
    raise FileNotFoundError(
        f"Final adapter not found: {ADAPTER_PATH}"
    )

model = load_adapter_for_inference(
    model_id=model_cfg["model_id"],
    adapter_path=ADAPTER_PATH,
    precision=model_cfg.get("precision", "bf16"),
    load_in_4bit=True,
)

model.eval()

model_device = next(model.parameters()).device

print("Model loaded successfully.")
print("Device:", model_device)


# ============================================================
# Display helpers
# ============================================================

DISPLAY_NAMES = {
    "crack": "Crack",
    "void": "Void",
    "honeycomb": "Honeycomb",
    "looseness": "Looseness",
    "spalling": "Spalling",
    "exposed_rebar": "Exposed Rebar",
    "corrosion": "Corrosion",
    "efflorescence": "Efflorescence",
    "pothole": "Pothole",
}


def display_name(category: str) -> str:
    return DISPLAY_NAMES.get(
        category,
        category.replace("_", " ").title(),
    )


def format_categories(categories: list[str]) -> str:
    if not categories:
        return "No valid damage category detected"
    return ", ".join(display_name(c) for c in categories)


# ------------------------------------------------------------
# HTML renderers for the results panel
# ------------------------------------------------------------

EMPTY_FINDINGS_HTML = """
<div class="findings findings--empty">
    <div class="findings-empty-mark"></div>
    <p>Findings appear here after you analyze an image.</p>
</div>
"""

EMPTY_STATUS_HTML = """
<div class="status status--idle">
    <span class="status-dot"></span>
    Ready
</div>
"""


def render_findings(categories: list[str]) -> str:
    if not categories:
        return """
        <div class="findings">
            <div class="findings-count">No damage found</div>
            <div class="tag-row">
                <span class="tag tag--none">
                    No valid damage category detected
                </span>
            </div>
        </div>
        """

    count = len(categories)
    noun = "finding" if count == 1 else "findings"

    tags = "".join(
        f'<span class="tag" style="--i:{index}">'
        f"{html.escape(display_name(category))}"
        f"</span>"
        for index, category in enumerate(categories)
    )

    return f"""
    <div class="findings">
        <div class="findings-count">
            <strong>{count}</strong> {noun}
        </div>
        <div class="tag-row">{tags}</div>
    </div>
    """


def render_status(kind: str, message: str) -> str:
    return f"""
    <div class="status status--{kind}">
        <span class="status-dot"></span>
        {html.escape(message)}
    </div>
    """


# ============================================================
# Inference function
# ============================================================

def analyze_image(image):

    if image is None:
        return (
            EMPTY_FINDINGS_HTML,
            "",
            render_status(
                "warn",
                "Upload an image first, then select Analyze damage.",
            ),
        )

    try:

        visual_inputs = build_visual_inputs(
            image,
            visual_input_mode=visual_input_mode,
            num_local_crops=num_local_crops,
            crop_fraction=crop_fraction,
        )

        messages = make_messages(
            visual_inputs,
            target=None,
            prompt_variant=prompt_variant,
        )

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        inputs = {
            key: (
                value.to(model_device)
                if torch.is_tensor(value)
                else value
            )
            for key, value in inputs.items()
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_seconds = time.perf_counter() - start_time

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generated[0, prompt_length:]

        raw_output = processor.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        categories, description, json_valid = parse_prediction(
            raw_output
        )

        if json_valid:
            status = render_status(
                "ok",
                f"Analyzed in {inference_seconds:.2f} s",
            )
        else:
            status = render_status(
                "warn",
                f"Analyzed in {inference_seconds:.2f} s. "
                f"The response wasn't in the expected format, "
                f"so categories may be incomplete.",
            )

        return (
            render_findings(categories),
            description,
            status,
        )

    except Exception as error:

        print("\nInference error:")
        print(error)

        return (
            EMPTY_FINDINGS_HTML,
            str(error),
            render_status(
                "error",
                "Analysis failed. See the terminal for the full error.",
            ),
        )


# ============================================================
# Styling
# ============================================================

# ------------------------------------------------------------
# Page background: textured concrete with a faint crack network,
# blueprint grid and warm/cool light pools.
# ------------------------------------------------------------

def svg_uri(svg: str) -> str:
    return 'url("data:image/svg+xml,' + quote(" ".join(svg.split())) + '")'


CONCRETE_NOISE = svg_uri("""
<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'>
  <filter id='n'>
    <feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/>
    <feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.07 0'/>
  </filter>
  <rect width='100%' height='100%' filter='url(#n)'/>
</svg>
""")

AGGREGATE = svg_uri("""
<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'>
  <g fill='#16263a' fill-opacity='0.07'>
    <ellipse cx='22' cy='30' rx='5' ry='3.5' transform='rotate(20 22 30)'/>
    <ellipse cx='96' cy='18' rx='3' ry='2'/>
    <ellipse cx='150' cy='62' rx='6' ry='4' transform='rotate(-30 150 62)'/>
    <ellipse cx='60' cy='98' rx='4' ry='2.6' transform='rotate(50 60 98)'/>
    <ellipse cx='124' cy='130' rx='2.6' ry='2'/>
    <ellipse cx='30' cy='150' rx='3.4' ry='2.4' transform='rotate(-15 30 150)'/>
    <ellipse cx='168' cy='166' rx='4.4' ry='3'/>
  </g>
</svg>
""")

CRACKS = svg_uri("""
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 1000' preserveAspectRatio='xMidYMid slice'>
  <g fill='none' stroke='#ffffff' stroke-linecap='round' stroke-linejoin='round'>
    <path stroke-opacity='0.45' stroke-width='2.6'
      d='M-10 560 L70 590 L110 580 L170 640 L210 636 L250 700 L320 716 L350 780 L420 800 L470 870 L540 900 L580 1010'/>
    <path stroke-opacity='0.32' stroke-width='1.4' d='M170 640 L160 700 L186 742'/>
    <path stroke-opacity='0.32' stroke-width='1.4' d='M320 716 L372 690 L410 700 L450 668'/>
    <path stroke-opacity='0.26' stroke-width='1' d='M420 800 L400 850 L416 890'/>

    <path stroke-opacity='0.42' stroke-width='2.4'
      d='M1610 80 L1540 110 L1500 100 L1450 160 L1400 170 L1370 230 L1300 250 L1280 320 L1220 350'/>
    <path stroke-opacity='0.30' stroke-width='1.3' d='M1450 160 L1470 220 L1450 262'/>
    <path stroke-opacity='0.30' stroke-width='1.3' d='M1300 250 L1250 226 L1210 236'/>

    <path stroke-opacity='0.38' stroke-width='2'
      d='M1610 760 L1560 770 L1520 820 L1470 830 L1440 880 L1380 900 L1360 1010'/>
    <path stroke-opacity='0.26' stroke-width='1.2' d='M1520 820 L1540 870 L1528 910'/>
  </g>
  <g fill='none' stroke='#ffe066' stroke-linecap='round' stroke-opacity='0.9' stroke-width='3'>
    <path d='M1236 330 q-14 30 6 52 q22 20 -2 46'/>
    <path d='M226 612 q18 26 -4 48'/>
  </g>
</svg>
""")

PAGE_BG = ", ".join([
    CRACKS + " center / cover no-repeat",
    CONCRETE_NOISE,
    "linear-gradient(rgba(255,255,255,0.07) 1px, transparent 1px) 0 0 / 56px 56px",
    "linear-gradient(90deg, rgba(255,255,255,0.07) 1px, transparent 1px) 0 0 / 56px 56px",
    "radial-gradient(60vw 55vh at 0% 0%, rgba(255,201,60,0.95), transparent 62%)",
    "radial-gradient(55vw 60vh at 100% 8%, rgba(63,208,201,0.85), transparent 60%)",
    "radial-gradient(60vw 60vh at 100% 100%, rgba(255,107,74,0.90), transparent 62%)",
    "radial-gradient(55vw 55vh at 0% 100%, rgba(108,92,231,0.75), transparent 60%)",
    "linear-gradient(135deg, #1d4f7a 0%, #1f7a8c 45%, #2a5d8f 100%)",
])


CSS = """

@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&display=swap');

:root {
    --steel:      #16263a;   /* deep structural steel */
    --steel-2:    #24405e;
    --concrete:   #e4e6e3;   /* poured concrete */
    --concrete-2: #f3f4f1;
    --marking:    #f5b400;   /* inspection marking paint */
    --rust:       #b0512a;   /* rebar corrosion */
    --ink:        #1b2530;
    --muted:      #5d6873;
    --line:       #cfd3cf;
    --font: 'Archivo', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
}

html, body { min-height: 100%; }

html,
body,
gradio-app {
    margin: 0 !important;
    min-height: 100vh !important;
    color: var(--ink);
    background: """ + PAGE_BG + """ !important;
    background-attachment: fixed !important;
}

/* Gradio paints white fills on its wrappers; clear them so the page shows */
:root, .dark {
    --body-background-fill: transparent !important;
}

gradio-app > div,
.gradio-container,
.gradio-container > .main,
.main,
.app,
.contain,
.fillable {
    background: transparent !important;
    background-color: transparent !important;
}

.gradio-container,
.gradio-container * {
    font-family: var(--font) !important;
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 28px 28px 40px !important;
    background: transparent !important;
}

footer { display: none !important; }


/* =========================================================
   Hero: the one bold moment
   ========================================================= */

#hero {
    position: relative;
    overflow: hidden;
    border-radius: 6px;
    padding: 46px 48px 64px;
    margin-bottom: 26px;
    color: #fff;
    background-color: var(--steel);
    /* blueprint grid */
    background-image:
        linear-gradient(rgba(255,255,255,0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.055) 1px, transparent 1px),
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 96px 96px, 96px 96px, 24px 24px, 24px 24px;
    box-shadow: 0 30px 60px -24px rgba(10,20,40,0.75), 0 0 0 1px rgba(255,255,255,0.12);
}

/* hazard tape along the bottom edge */
#hero::after {
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 18px;
    background: repeating-linear-gradient(
        -45deg,
        var(--marking) 0 18px,
        var(--steel) 18px 36px
    );
}

/* hand-drawn crack line across the grid */
#hero .crack {
    position: absolute;
    right: 40px;
    top: 50%;
    width: 260px;
    height: 150px;
    transform: translateY(-62%);
    opacity: 0.9;
    pointer-events: none;
}

#hero .crack path {
    fill: none;
    stroke: var(--marking);
    stroke-width: 2.4;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 600;
    stroke-dashoffset: 600;
    animation: draw-crack 1.6s 0.2s cubic-bezier(.6,.05,.3,1) forwards;
}

#hero .crack path.branch {
    stroke-width: 1.4;
    animation-delay: 0.9s;
}

@keyframes draw-crack {
    to { stroke-dashoffset: 0; }
}

#hero h1 {
    position: relative;
    z-index: 2;
    margin: 0 !important;
    max-width: 900px;
    color: #fff !important;
    font-size: clamp(28px, 3.4vw, 46px) !important;
    line-height: 1.08 !important;
    font-weight: 850 !important;
    font-stretch: 118%;
    letter-spacing: 0.005em;
}



/* =========================================================
   Panels
   ========================================================= */

.panel-card {
    background: rgba(255, 255, 255, 0.90) !important;
    backdrop-filter: blur(10px) saturate(1.1);
    -webkit-backdrop-filter: blur(10px) saturate(1.1);
    border: 1px solid rgba(255, 255, 255, 0.7) !important;
    outline: 1px solid rgba(22, 38, 58, 0.10);
    border-radius: 6px !important;
    padding: 24px !important;
    box-shadow: 0 1px 0 #fff inset, 0 28px 56px -26px rgba(10,20,40,0.65) !important;
}

.panel-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}

.panel-head .num {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    flex: none;
    border-radius: 50%;
    background: var(--steel);
    color: var(--marking);
    font-size: 14px;
    font-weight: 800;
}

.panel-head h2 {
    margin: 0 !important;
    color: var(--steel) !important;
    font-size: 21px !important;
    font-weight: 800 !important;
    font-stretch: 112%;
}

.panel-copy {
    margin: 0 0 16px 42px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.5;
}

.field-label {
    display: block;
    margin: 18px 0 8px;
    color: var(--steel);
    font-size: 14px;
    font-weight: 700;
}


/* =========================================================
   Image drop zone
   ========================================================= */

#input-image {
    border-radius: 4px !important;
    border: 2px dashed #b7bdb8 !important;
    background: #fbfbf9 !important;
    overflow: hidden !important;
}

#input-image:hover {
    border-color: var(--steel-2) !important;
}


/* =========================================================
   Buttons
   ========================================================= */

#analyze-btn {
    min-height: 54px !important;
    border: none !important;
    border-radius: 4px !important;
    background: var(--marking) !important;
    color: var(--steel) !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    font-stretch: 112%;
    box-shadow: 0 3px 0 #b88700 !important;
    transition: transform .12s ease, box-shadow .12s ease, filter .12s ease !important;
}

#analyze-btn:hover  { filter: brightness(1.05); }
#analyze-btn:active { transform: translateY(3px); box-shadow: 0 0 0 #b88700 !important; }

#clear-btn {
    min-height: 54px !important;
    border-radius: 4px !important;
    background: transparent !important;
    color: var(--steel) !important;
    font-weight: 700 !important;
    border: 1.5px solid var(--steel) !important;
    box-shadow: none !important;
}

#clear-btn:hover { background: rgba(22,38,58,0.06) !important; }

#analyze-btn:focus-visible,
#clear-btn:focus-visible {
    outline: 3px solid var(--steel-2) !important;
    outline-offset: 2px;
}


/* =========================================================
   Findings (category tags)
   ========================================================= */

#findings-box, #status-box { padding: 0 !important; }

.findings {
    min-height: 118px;
    padding: 18px;
    border-radius: 4px;
    background: var(--steel);
    color: #fff;
}

.findings-count {
    margin-bottom: 12px;
    color: #9fb2c6;
    font-size: 14px;
}

.findings-count strong {
    color: #fff;
    font-size: 26px;
    font-weight: 850;
    margin-right: 4px;
}

.tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.tag {
    display: inline-block;
    padding: 9px 16px;
    border-radius: 3px;
    background: var(--marking);
    color: var(--steel);
    font-size: 16px;
    font-weight: 800;
    font-stretch: 110%;
    transform-origin: left center;
    animation: stamp .38s calc(var(--i, 0) * 90ms) cubic-bezier(.2,1.4,.4,1) both;
}

.tag--none {
    background: #3a5068;
    color: #dfe7ef;
    animation: none;
}

@keyframes stamp {
    from { opacity: 0; transform: scale(1.35) rotate(-2deg); }
    to   { opacity: 1; transform: scale(1) rotate(0); }
}

.findings--empty {
    display: flex;
    align-items: center;
    gap: 14px;
    background: repeating-linear-gradient(
        -45deg,
        #e9ebe7 0 10px,
        #f1f2ef 10px 20px
    );
    border: 1px solid var(--line);
    color: var(--muted);
}

.findings--empty p { margin: 0; font-size: 15px; }

.findings-empty-mark {
    width: 34px;
    height: 34px;
    flex: none;
    border: 2px dashed #a5aca7;
    border-radius: 50%;
}


/* =========================================================
   Description + status
   ========================================================= */

#description-box textarea {
    border-radius: 4px !important;
    border: 1px solid var(--line) !important;
    border-left: 4px solid var(--rust) !important;
    background: #fff !important;
    color: var(--ink) !important;
    font-size: 15.5px !important;
    line-height: 1.6 !important;
    padding: 14px 16px !important;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    margin-top: 6px;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
    line-height: 1.4;
}

.status-dot {
    width: 9px;
    height: 9px;
    flex: none;
    border-radius: 50%;
    background: currentColor;
}

.status--idle  { background: #e8eae6; color: var(--muted); }
.status--ok    { background: #e1efe6; color: #1f6b43; }
.status--warn  { background: #fbefd2; color: #7a5600; }
.status--error { background: #f6e0d8; color: #9a3412; }


/* =========================================================
   Responsive + reduced motion
   ========================================================= */

@media (max-width: 800px) {
    .gradio-container { padding: 14px !important; }
    #hero { padding: 30px 24px 54px; }
    #hero .crack { opacity: 0.35; right: -60px; }
    .panel-card { padding: 16px !important; }
    .panel-copy { margin-left: 0; }
}

@media (prefers-reduced-motion: reduce) {
    #hero .crack path { animation: none; stroke-dashoffset: 0; }
    .tag { animation: none; }
}
"""

# Keep the page in its light, concrete look even when the OS is in dark mode.
FORCE_LIGHT_JS = """
() => {
    document.body.classList.remove('dark');
    document.documentElement.classList.remove('dark');
    let el = document.querySelector('.gradio-container');
    while (el && el !== document.body) {
        el.style.setProperty('background', 'transparent', 'important');
        el = el.parentElement;
    }
}
"""


# ============================================================
# Interface
# ============================================================

HERO_HTML = f"""
<div id="hero">
    <svg class="crack" viewBox="0 0 340 200" aria-hidden="true">
        <path d="M6 18 L58 44 L84 38 L122 78 L150 74 L176 112
                 L214 118 L236 150 L282 160 L334 194" />
        <path class="branch" d="M122 78 L118 112 L134 136" />
        <path class="branch" d="M214 118 L246 104 L270 110" />
    </svg>

    <h1>Structural Damage Intelligence</h1>

</div>
"""


with gr.Blocks(
    title="Structural Damage Intelligence",
    css=CSS,
    js=FORCE_LIGHT_JS,
) as demo:

    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=True):

        # ---------------- Left: upload ----------------
        with gr.Column(scale=5, elem_classes=["panel-card"]):

            gr.HTML(
                """
                <div class="panel-head">
                    <span class="num">1</span>
                    <h2>Upload an image</h2>
                </div>
                <p class="panel-copy">
                    Drop a photo here or click to browse.
                    Close, well-lit shots give the best results.
                </p>
                """
            )

            input_image = gr.Image(
                type="pil",
                show_label=False,
                height=440,
                elem_id="input-image",
            )

            with gr.Row():
                analyze_button = gr.Button(
                    "Analyze damage",
                    variant="primary",
                    size="lg",
                    elem_id="analyze-btn",
                    scale=3,
                )
                clear_button = gr.ClearButton(
                    value="Clear",
                    elem_id="clear-btn",
                    scale=1,
                )

        # ---------------- Right: results ----------------
        with gr.Column(scale=6, elem_classes=["panel-card"]):

            gr.HTML(
                """
                <div class="panel-head">
                    <span class="num">2</span>
                    <h2>Review the assessment</h2>
                </div>
                <p class="panel-copy">
                    Damage types found in the image, followed by
                    a written description.
                </p>
                """
            )

            gr.HTML('<span class="field-label">Damage found</span>')

            damage_output = gr.HTML(
                EMPTY_FINDINGS_HTML,
                elem_id="findings-box",
            )

            gr.HTML('<span class="field-label">Inspection note</span>')

            description_output = gr.Textbox(
                show_label=False,
                interactive=False,
                lines=7,
                elem_id="description-box",
                placeholder="The written description appears here.",
            )

            status_output = gr.HTML(
                EMPTY_STATUS_HTML,
                elem_id="status-box",
            )

    # ---------------- Actions ----------------

    analyze_button.click(
        fn=analyze_image,
        inputs=[input_image],
        outputs=[damage_output, description_output, status_output],
    )

    # Clear resets the image and every result field.
    clear_button.add([input_image, description_output])
    clear_button.click(
        fn=lambda: (EMPTY_FINDINGS_HTML, EMPTY_STATUS_HTML),
        inputs=None,
        outputs=[damage_output, status_output],
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        inbrowser=True,
        show_error=True,
    )