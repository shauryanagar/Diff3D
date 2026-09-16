import os
import sys


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["WEBRTC_ENABLED"] = "false"
os.environ["MPLBACKEND"] = "Agg"

import open3d as o3d
import torch
import gradio as gr


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from src.model import CondPointDiffusionUNet
from src.sampler import sample_ddim
from src.mesh import points_to_subdivided_stl

CLASSES = ["cone", "cube", "cylinder", "pencil", "plane", "sphere", "table", "vase"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}


device = torch.device("cpu")
print(f"Diff3D backend initialized on: {device}")

model = CondPointDiffusionUNet(num_classes=8).to(device)

WEIGHTS_PATH = "checkpoints/model_weights.pt"
if os.path.exists(WEIGHTS_PATH):
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    cleaned_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_dict)
    model.eval()
    print("Checkpoints loaded successfully.")
else:
    print(f"Warning: Model weights not found at '{WEIGHTS_PATH}'")

def generate_3d_object(prompt_word, ddim_steps):
    prompt_clean = prompt_word.strip().lower()
    if prompt_clean not in CLASS_TO_IDX:
        raise gr.Error(f"Unsupported class: '{prompt_word}'")

    idx = CLASS_TO_IDX[prompt_clean]
    
    # 1. Deterministic DDIM trajectory on CPU
    pts = sample_ddim(model, class_idx=idx, steps=int(ddim_steps), device=device)
    
    # 2. Watertight Poisson surface reconstruction
    out_file = f"output_{prompt_clean}.stl"
    points_to_subdivided_stl(pts, out_file)
    return out_file

with gr.Blocks(title="Diff3D") as demo:
    gr.Markdown("# Diff3D")
    gr.Markdown(" 3D mesh generation via DDIM and surface reconstruction of point clouds")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Dropdown(
                choices=CLASSES,
                value="vase",
                label="One-Word Text Prompt"
            )
            steps_slider = gr.Slider(
                minimum=0, 
                maximum=80, 
                value=40, 
                step=10, 
                label="DDIM Inference Steps"
            )
            generate_btn = gr.Button("Generate 3D Model", variant="primary")

        with gr.Column():
            viewer = gr.Model3D(label="Interactive 3D STL Preview")

    generate_btn.click(
        fn=generate_3d_object,
        inputs=[prompt_input, steps_slider],
        outputs=[viewer]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        inbrowser=False
    )