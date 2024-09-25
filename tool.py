import pandas as pd
import os
import yaml
from tqdm import tqdm

config_data = {
    'txt2img': {
        'sd_model_key': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
        'ip_adapter_image_path': None,
        'prompt': 'turn around, monkey head, (Sci-Fi digital painting:1.5), colorful, painting, high quality',
        'negative_prompt': 'strong light, Bright light, intense light, dazzling light, brilliant light, radiant light, Shade, darkness, silhouette, dimness, obscurity, shadow, glasses',
        'seed': 1713428430,
        'width': 1024,
        'height': 512,
        'num_images_per_prompt': 1,
        'guidance_scale': 7.0,
        'num_inference_steps': 20,
        'controlnet_units': [
            {
                'preprocessor': 'none',
                'controlnet_key': 'lllyasviel/control_v11f1p_sd15_depth',
                'condition_image_path': None,
                'weight': 1.0
            }
        ]
    },
    'inpaint': {
        'sd_model_key': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
        'image_path': None,
        'mask_path': None,
        'ip_adapter_image_path': None,
        'prompt': None,
        'negative_prompt': None,
        'seed': None,
        'width': 1024,
        'height': 512,
        'num_images_per_prompt': 1,
        'guidance_scale': 3.0,
        'num_inference_steps': 20,
        'denoising_strength': 1.0,
        'controlnet_units': [
            {
                'preprocessor': 'none',
                'controlnet_key': 'lllyasviel/control_v11f1p_sd15_depth',
                'condition_image_path': None,
                'weight': 1.0
            },
            {
                'preprocessor': 'inpaint_global_harmonious',
                'controlnet_key': 'lllyasviel/control_v11p_sd15_inpaint',
                'condition_image_path': '',
                'weight': 0.5
            }
        ]
    }
}

def main():
    objects_info_path = "ExportedOBJ_List2.csv"
    df = pd.read_csv(objects_info_path)
    uid_list = df['uid'].tolist()
    uid_to_name = dict(zip(df['uid'], df['name']))
    uid_to_description = dict(zip(df['uid'], df['description']))

    for uid in tqdm(uid_list):
        name = uid_to_name[uid]
        description = uid_to_description[uid]
        obj_path = f"Objects/{name}.obj"
        target_path = f"Objects/{uid}/model.obj"
        config_path = f"Objects/{uid}/config.yaml"

        if not os.path.exists(obj_path):
            print(f"Obj file missing: {obj_path}")
            continue

        os.makedirs(f"Objects/{uid}", exist_ok=True)

        config_data['txt2img']['prompt'] = f"{description}, high quality"

        with open(config_path, 'w') as yaml_file:
            yaml.dump(config_data, yaml_file, default_flow_style=False)
        os.system(f"mv {obj_path} {target_path}")

        

if __name__ == "__main__":
    main()