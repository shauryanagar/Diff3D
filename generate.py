import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["WEBRTC_ENABLED"] = "false"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import open3d as o3d
import torch

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from src.model import CondPointDiffusionUNet
from src.sampler import sample_ddim
from src.mesh import points_to_subdivided_stl

CLASSES = ["cone", "cube", "cylinder", "pencil", "plane", "sphere", "table", "vase"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}

def main():
    parser = argparse.ArgumentParser(description="Diff3D: CLI Mesh Generation")
    parser.add_argument("--prompt", type=str, default="vase", choices=CLASSES, help="Archetype prompt.")
    parser.add_argument("--steps", type=int, default=40, help="DDIM inference steps.")
    parser.add_argument("--out", type=str, default=None, help="Output .stl filename.")
    parser.add_argument("--weights", type=str, default="checkpoints/model_weights.pt", help="Path to weights.")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Diff3D running CLI on: {device}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint not found at '{args.weights}'.")

    model = CondPointDiffusionUNet(num_classes=8).to(device)
    state_dict = torch.load(args.weights, map_location=device)
    cleaned_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_dict)
    model.eval()

    output_path = args.out if args.out else f"{args.prompt}.stl"
    class_idx = CLASS_TO_IDX[args.prompt.lower()]

    print(f"Sampling '{args.prompt}' across {args.steps} DDIM steps...")
    pts = sample_ddim(model, class_idx=class_idx, steps=args.steps, device=device)

    print("Reconstructing watertight mesh...")
    points_to_subdivided_stl(pts, output_path)
    print(f"Success: Saved to {output_path}")

if __name__ == "__main__":
    main()