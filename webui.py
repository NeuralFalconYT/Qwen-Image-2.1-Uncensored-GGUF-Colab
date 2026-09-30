import asyncio
import gradio as gr

import qwen_text_to_image as qwen


# ============================================================
# LOAD MODEL ONCE
# ============================================================

asyncio.run(qwen.load_model())


# ============================================================
# RESOLUTION PRESETS
# ============================================================

ASPECT_PRESETS = {
    "1:1  •  640 × 640": (640, 640),
    "16:9  •  768 × 432": (768, 432),
    "9:16  •  432 × 768": (432, 768),
    "4:3  •  704 × 528": (704, 528),
    "3:4  •  528 × 704": (528, 704),
    "3:2  •  720 × 480": (720, 480),
    "2:3  •  480 × 720": (480, 720),
    "9:7  •  672 × 512": (672, 512),
    "7:9  •  512 × 672": (512, 672),
    "21:9  •  768 × 336": (768, 336),
    "9:21  •  336 × 768": (336, 768),
}

CUSTOM_ASPECT = "Custom"

ASPECT_CHOICES = list(ASPECT_PRESETS.keys()) + [CUSTOM_ASPECT]

DEFAULT_ASPECT = "16:9  •  768 × 432"
DEFAULT_WIDTH, DEFAULT_HEIGHT = ASPECT_PRESETS[DEFAULT_ASPECT]

DEFAULT_NEGATIVE = (
    "blurry, low quality, bad anatomy, deformed hands, extra fingers, "
    "bad proportions, distorted face, duplicate body parts, lowres, "
    "watermark, text, logo"
)


# ============================================================
# UI HELPERS
# ============================================================

def apply_aspect_ratio(choice):
    """
    Selecting a preset updates Width + Height.
    Custom keeps the current values untouched.
    """
    if choice == CUSTOM_ASPECT:
        return gr.skip(), gr.skip()

    width, height = ASPECT_PRESETS[choice]

    return (
        gr.update(value=width),
        gr.update(value=height),
    )


def mark_as_custom(_value):
    """
    If user manually edits Width or Height, switch Aspect Ratio to Custom.
    """
    return gr.update(value=CUSTOM_ASPECT)


def toggle_seed(randomize):
    return gr.update(visible=not randomize)


# ============================================================
# GENERATION
# ============================================================

def generate_image(
    prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    denoise,
    sampler_name,
    scheduler,
    seed,
    randomize_seed,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    width = int(width)
    height = int(height)

    if width < 256 or height < 256:
        raise gr.Error("Width and height must be at least 256.")

    # qwen_text_to_image.py aligns dimensions to multiples of 16.
    seed_value = None if randomize_seed else int(seed)

    result = qwen.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=int(steps),
        cfg=float(cfg),
        denoise=float(denoise),
        sampler_name=sampler_name,
        scheduler=scheduler,
        seed=seed_value,
        output_dir="/content/saved_images",
        cleanup_after=True,
        verbose=True,
    )

    return (
        result["image"],
        result["seed"],
    )


# ============================================================
# EXAMPLES
# ============================================================

examples = [
    [
        "A beautiful anime-style young woman lying peacefully on lush green grass "
        "in a wide mountain meadow, distant majestic mountains, soft sunlight, "
        "gentle breeze moving her hair, cinematic composition, detailed face, "
        "dreamy atmosphere"
    ],
    [
        "A cinematic futuristic anime warrior standing on a rooftop at night, "
        "neon city skyline, rain, dramatic rim lighting, black coat moving in "
        "the wind, premium movie poster composition"
    ],
    [
        "A majestic fantasy dragon flying above snowy mountains during sunset, "
        "cinematic clouds, dramatic lighting, ultra detailed scales, epic fantasy artwork"
    ],
    [
        "A futuristic cyberpunk city at night, glowing neon signs, wet streets, "
        "flying vehicles, atmospheric fog, cinematic science fiction photography"
    ],
]


# ============================================================
# THEME
# ============================================================

custom_theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="blue",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
    text_size="lg",
    spacing_size="md",
    radius_size="lg",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    block_title_text_weight="600",
)


CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

.header-text {
    text-align: center !important;
}

.header-text h1 {
    text-align: center !important;
    font-size: 2.5rem !important;
    font-weight: 750 !important;
    margin-bottom: 0.35rem !important;
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.header-text p {
    text-align: center !important;
    font-size: 1.05rem !important;
    color: #64748b !important;
    margin-top: 0 !important;
}

.generate-button {
    min-height: 50px !important;
    font-weight: 700 !important;
}

button, .gr-button {
    transition: all 0.2s ease !important;
}

button:hover, .gr-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
}

@media (max-width: 768px) {
    .header-text h1 {
        font-size: 1.8rem !important;
    }

    .header-text p {
        font-size: 0.95rem !important;
    }
}
"""


# ============================================================
# WEB UI
# ============================================================

with gr.Blocks(fill_height=True) as demo:

    gr.Markdown(
        """
# 🎨 Qwen Image 2.1 Uncensored GGUF
**@NeuralFalcon**
        """,
        elem_classes="header-text",
    )

    with gr.Row(equal_height=False):

        # ====================================================
        # LEFT COLUMN
        # ====================================================

        with gr.Column(scale=1, min_width=320):

            prompt = gr.Textbox(
                label="✨ Your Prompt",
                placeholder="Describe the image you want to create...",
                lines=5,
                max_lines=10,
                autofocus=True,
            )


            # Main/simple setting stays OUTSIDE Advanced Settings.
            aspect_ratio = gr.Dropdown(
                choices=ASPECT_CHOICES,
                value=DEFAULT_ASPECT,
                label="🖼️ Aspect Ratio",
                info="Choose a preset, or manually change Width / Height below.",
            )

            # =================================================
            # ADVANCED SETTINGS
            # =================================================

            with gr.Accordion("⚙️ Advanced Settings", open=False):

                gr.Markdown("#### Resolution")

                with gr.Row():
                    width = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        value=DEFAULT_WIDTH,
                        step=16,
                        label="Width",
                        info="Manual override allowed",
                    )

                    height = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        value=DEFAULT_HEIGHT,
                        step=16,
                        label="Height",
                        info="Manual override allowed",
                    )

                gr.Markdown("#### Generation")

                steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=1,
                    label="Inference Steps",
                    info="More steps = slower generation",
                )

                with gr.Row():
                    cfg = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=2.2,
                        step=0.1,
                        label="CFG",
                    )

                    denoise = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=1.0,
                        step=0.01,
                        label="Denoise",
                    )

                with gr.Row():
                    sampler_name = gr.Dropdown(
                        choices=[
                            "euler",
                            "euler_ancestral",
                            "heun",
                            "ddim",
                            "dpmpp_2m",
                            "dpmpp_2m_sde",
                        ],
                        value="euler",
                        label="Sampler",
                    )

                    scheduler = gr.Dropdown(
                        choices=[
                            "simple",
                            "normal",
                            "karras",
                            "exponential",
                            "sgm_uniform",
                        ],
                        value="simple",
                        label="Scheduler",
                    )

                gr.Markdown("#### Seed")

                with gr.Row():
                    randomize_seed = gr.Checkbox(
                        label="🎲 Random Seed",
                        value=True,
                    )

                    seed = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0,
                        visible=False,
                    )
                negative_prompt = gr.Textbox(
                    label="🚫 Negative Prompt",
                    value=DEFAULT_NEGATIVE,
                    lines=3,
                    )

                randomize_seed.change(
                    fn=toggle_seed,
                    inputs=[randomize_seed],
                    outputs=[seed],
                )

            # Aspect preset -> automatically update advanced Width + Height
            aspect_ratio.change(
                fn=apply_aspect_ratio,
                inputs=[aspect_ratio],
                outputs=[width, height],
            )

            # Manual Width/Height edit -> Aspect Ratio becomes Custom.
            # .input = only user interaction, so preset callbacks do not fight this.
            width.input(
                fn=mark_as_custom,
                inputs=[width],
                outputs=[aspect_ratio],
            )

            height.input(
                fn=mark_as_custom,
                inputs=[height],
                outputs=[aspect_ratio],
            )

            generate_btn = gr.Button(
                "🚀 Generate Image",
                variant="primary",
                size="lg",
                scale=1,
                elem_classes="generate-button",
            )

            gr.Examples(
                examples=examples,
                inputs=[prompt],
                label="💡 Try these prompts",
                examples_per_page=4,
            )

        # ====================================================
        # RIGHT COLUMN
        # ====================================================

        with gr.Column(scale=1, min_width=320):

            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                format="png",
                show_label=False,
                height=600,
                buttons=["download", "share"],
            )

            used_seed = gr.Number(
                label="🎲 Seed Used",
                interactive=False,
                container=True,
            )

    # ========================================================
    # GENERATE EVENTS
    # ========================================================

    generation_inputs = [
        prompt,
        negative_prompt,
        width,
        height,
        steps,
        cfg,
        denoise,
        sampler_name,
        scheduler,
        seed,
        randomize_seed,
    ]

    generation_outputs = [
        output_image,
        used_seed,
    ]

    generate_btn.click(
        fn=generate_image,
        inputs=generation_inputs,
        outputs=generation_outputs,
    )

    prompt.submit(
        fn=generate_image,
        inputs=generation_inputs,
        outputs=generation_outputs,
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)

    demo.launch(
        theme=custom_theme,
        css=CSS,
        footer_links=["api", "gradio"],
        mcp_server=True,
        debug=True,
        share=True
    )
