import pandas as pd
import subprocess
import os

import yaml
from tqdm import tqdm

style_prompt = "christmas style"

def main():
    # Read the CSV and extract the list of uids
    df = pd.read_csv('Objects.csv')
    uid_list = df['uid'].tolist()

    # Loop through the uid list and run the experiment for each uid
    for uid in tqdm(uid_list):
        config_path = f"objaverse/{uid}/config.yaml"
        mesh_path = f"objaverse/{uid}/model.obj"
        output_path = f"objaverse/{uid}/output"
        render_path = "paint3d/config/train_config_paint3d.py"
        updated_config_path = f"objaverse/{uid}/runtime_config.yaml"
        
        if not os.path.exists(config_path):
            print(f"Config file missing: {config_path}")
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
        
        # Construct the command
        command = [
            "python", "pipeline_paint3d_stage1.py",
            "--sd_config", config_path,
            "--render_config", render_path,
            "--mesh_path", mesh_path,
            "--outdir", output_path
        ]
        
        # Run the command
        subprocess.run(command)

if __name__ == '__main__':
    main()
