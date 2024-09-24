import pandas as pd
import subprocess
import os

from tqdm import tqdm

max_hits = 1
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
        
        if not os.path.exists(config_path):
            print(f"Config file missing: {config_path}")
            continue
        
        os.makedirs(output_path, exist_ok=True)
        
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
