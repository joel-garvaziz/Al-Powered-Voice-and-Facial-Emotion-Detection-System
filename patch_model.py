import json
import zipfile
import shutil
import os

def patch_keras_file(file_path):
    print(f"Patching {file_path}...")
    temp_dir = 'temp_keras_patch'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    with zipfile.ZipFile(file_path, 'r') as zf:
        zf.extractall(temp_dir)
        
    config_path = os.path.join(temp_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Recursively remove 'quantization_config'
        def remove_quant(obj):
            if isinstance(obj, dict):
                obj.pop('quantization_config', None)
                for k, v in obj.items():
                    remove_quant(v)
            elif isinstance(obj, list):
                for v in obj:
                    remove_quant(v)
                    
        remove_quant(config)
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        # Re-zip the file
        with zipfile.ZipFile(file_path, 'w') as zf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, temp_dir)
                    zf.write(abs_path, arcname=rel_path)
        print("Success!")
    else:
        print("No config.json found. Skipping.")
        
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

patch_keras_file(r"d:\S6 Mini Project\Facial Detection\emotion_model.keras")
patch_keras_file(r"d:\S6 Mini Project\HuBERT Model\voice_emotion_detection_hubert_large.keras")
