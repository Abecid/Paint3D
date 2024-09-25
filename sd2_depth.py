import requests
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

import torch
from diffusers import StableDiffusionDepth2ImgPipeline
import trimesh
import pyrender

os.environ["PYOPENGL_PLATFORM"] = "egl"

style_prompt = "christmas style"

def render_mesh_to_image(obj_path, azimuth=0, elevation=0, distance=2):
    """Render an OBJ file using pyrender and return the rendered image."""
    # Load the 3D model
    mesh = trimesh.load(obj_path)

    # Check if we have a Trimesh object or a Scene
    if isinstance(mesh, trimesh.Scene):
        # If it's a Scene, combine all geometries into a single Trimesh object
        mesh = trimesh.util.concatenate(mesh.dump())

    # Set up the scene
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)

    # Set up the camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(elevation), -np.sin(elevation), distance],
        [0.0, np.sin(elevation), np.cos(elevation), distance],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    # Set up the light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)

    # Convert the rendered image to a PIL image
    rendered_image = Image.fromarray(color)
    return rendered_image

def main():
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth",
        torch_dtype=torch.float16,
    ).to("cuda")

    objects_info_path = "ExportedOBJ_List2.csv"
    df = pd.read_csv(objects_info_path)
    uid_list = df['uid'].tolist()
    uid_to_name = dict(zip(df['uid'], df['name']))
    uid_to_description = dict(zip(df['uid'], df['description']))

    for uid in tqdm(uid_list):
        name = uid_to_name[uid]
        description = uid_to_description[uid]
        obj_path = f"Objects/{uid}/model.obj"
        config_path = f"Objects/{uid}/config.yaml"

        if not os.path.exists(obj_path):
            print(f"Obj file missing: {obj_path}")
            continue
            
        init_image = render_mesh_to_image(obj_path, azimuth=0, elevation=0)

        # try:
        #     init_image = render_mesh_to_image(obj_path, azimuth=0, elevation=0)
        # except Exception as e:
        #     print(f"Error rendering image for {name}: {e}")
        #     continue
        # else:
        #     print(f"Rendered image for {name}")

        # Run the Stable Diffusion depth-to-image pipeline
        prompt = f"3D object of {description}"
        negative_prompt = "bad, deformed, ugly, bad anatomy"
        generated_image = pipe(
            prompt=prompt,
            image=init_image,
            negative_prompt=negative_prompt,
            strength=0.7
        ).images[0]

        # Save the generated image
        output_path = f"Objects/{uid}/sd2_dpeth.png"
        generated_image.save(output_path)
        print(f"Generated image saved to {output_path}")


if __name__ == "__main__":
    main()