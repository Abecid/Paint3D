import pandas as pd
import subprocess
import os

import yaml
from tqdm import tqdm

style_prompt = "christmas style"

def main():
    # Read the CSV and extract the list of uids
    df = pd.read_csv('Objects.csv')
    # df = pd.read_csv('ExportedOBJ_List2.csv')
    uid_list = df['uid'].tolist()
    
    render_path = "paint3d/config/train_config_paint3d.py"

    # Loop through the uid list and run the experiment for each uid
    for uid in tqdm(uid_list):
        config_path = f"objaverse/{uid}/config.yaml"
        mesh_path = f"objaverse/{uid}/model.obj"
        output_path = f"objaverse/{uid}/output"
        output_path_stage_2 = f"objaverse/{uid}/stage_2"
        updated_config_path = f"objaverse/{uid}/runtime_config.yaml"
        uv_config_path = "objaverse/{uid}/uv_config.yaml"
        updated_uv_config_path = "objaverse/{uid}/runtime_uv_config.yaml"
        ip_image_path = "demo/objs/Suzanne_monkey/img_prompt.png"
        
        # config_path = f"Objects/{uid}/config.yaml"
        # mesh_path = f"Objects/{uid}/model.obj"
        # output_path = f"Objects/{uid}/output"
        # updated_config_path = f"Objects/{uid}/runtime_config.yaml"
        
        if not os.path.exists(config_path) or not os.path.exists(mesh_path):
            print(f"Missing: {mesh_path}")
            continue
        
        os.makedirs(output_path, exist_ok=True)

        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)

        if 'txt2img' in config_data and 'prompt' in config_data['txt2img']:
            config_data['txt2img']['prompt'] += f" {style_prompt}"
        else:
            print(f"'txt2img' or 'prompt' not found in config: {config_path}")
            continue
        
        # Write the updated config to a new runtime config file
        with open(updated_config_path, 'w') as file:
            yaml.dump(config_data, file)
            
        with open(uv_config_path, 'r') as file:
            uv_config_data = yaml.safe_load(file)
            
        uv_config_data['inpaint']['prompt'] += f" {style_prompt}"
        uv_config_data['img2img']['prompt'] += f" {style_prompt}"
        
        with open(updated_uv_config_path, 'w') as file:
            yaml.dump(uv_config_data, file)
        
        # Construct the command
        command = [
            "python3", "pipeline_paint3d_stage1.py",
            "--sd_config", updated_config_path,
            "--render_config", render_path,
            "--mesh_path", mesh_path,
            "--outdir", output_path
        ]
        
        # Run the command
        subprocess.run(command)
        
        # Stage 2 Command
        command = [
            "python3", "pipeline_paint3d_stage2.py", 
            "--sd_config", updated_uv_config_path,
            "--render_config", render_path,
            "--mesh_path", mesh_path,
            "--texture_path", f"{output_path}/res-0/albedo.png",
            "--prompt", " ",
            "--ip_adapter_image_path", ip_image_path,
            "--outdir", output_path_stage_2
        ]
        
        subprocess.run(command)

if __name__ == '__main__':
    main()
