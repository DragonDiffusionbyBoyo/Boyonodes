import os
import json
import requests
import git
import threading
import sys
import subprocess
from typing import Dict, Any, List, Tuple

class BoyoAssetGrabberAdvanced:
    def __init__(self):
        self.downloading = False
        self.progress_message = ""
        self.pip_path = None
        self.python_env = None
        
    @classmethod
    def INPUT_TYPES(cls):
        # Get Boyonodes directory and look for assetJsons folder within it
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            boyonodes_dir = current_dir  # We're already in the Boyonodes directory
            asset_json_dir = os.path.join(boyonodes_dir, "assetJsons")
            
            # Create folder if it doesn't exist
            os.makedirs(asset_json_dir, exist_ok=True)
            
            # Scan for JSON files
            json_files = []
            if os.path.exists(asset_json_dir):
                for f in os.listdir(asset_json_dir):
                    if os.path.isfile(os.path.join(asset_json_dir, f)) and f.lower().endswith('.json'):
                        json_files.append(f)
            
            if not json_files:
                json_files = ["No JSON files found - add files to custom_nodes/Boyonodes/assetJsons/"]
                
        except Exception as e:
            json_files = [f"Error scanning folder: {str(e)}"]
        
        return {
            "required": {
                "json_file": (sorted(json_files),),
                "custom_nodes_path": ("STRING", {
                    "multiline": False,
                    "default": "C:\\Users\\YourName\\Documents\\ComfyUI",
                    "placeholder": "Path to your ComfyUI installation (for custom nodes)"
                }),
                "models_path": ("STRING", {
                    "multiline": False,
                    "default": "C:\\Users\\YourName\\Documents\\ComfyUI",
                    "placeholder": "Path to your ComfyUI installation (for models)"
                }),
                "download_trigger": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Download Assets",
                    "label_off": "Ready"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "download_assets"
    CATEGORY = "Boyo/Asset Management"
    
    def download_assets(self, json_file: str, custom_nodes_path: str, 
                       models_path: str, download_trigger: bool) -> Tuple[str]:
        if not download_trigger:
            return ("Ready to download assets",)
            
        if self.downloading:
            return (f"Currently downloading: {self.progress_message}",)
            
        try:
            # Handle dropdown selection - construct full path
            if not json_file or json_file.strip() == "":
                return ("ERROR: No JSON file selected",)
                
            if json_file.startswith("No JSON files found") or json_file.startswith("Error scanning"):
                return ("ERROR: " + json_file,)
            
            # Get Boyonodes directory and construct path to JSON file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            asset_json_dir = os.path.join(current_dir, "assetJsons")
            json_file_path = os.path.join(asset_json_dir, json_file)
            
            if not os.path.exists(json_file_path):
                return (f"ERROR: JSON file not found at {json_file_path}",)
                
            with open(json_file_path, 'r') as f:
                manifest = json.load(f)
                
            # Validate paths exist
            if not os.path.exists(custom_nodes_path):
                return (f"ERROR: Custom nodes path not found: {custom_nodes_path}",)
            if not os.path.exists(models_path):
                return (f"ERROR: Models path not found: {models_path}",)
                
            self.downloading = True
            status = self._process_manifest(manifest, custom_nodes_path, models_path)
            self.downloading = False
            
            return (status,)
            
        except Exception as e:
            self.downloading = False
            return (f"ERROR: {str(e)}",)
    
    def _process_manifest(self, manifest: Dict[str, Any], custom_nodes_path: str, 
                         models_path: str) -> str:
        """Process the asset manifest and download required files"""
        results = []
        
        # Detect Python environment and pip
        print(f"[BoyoAssetGrabber] Starting asset processing...")
        self.pip_path, self.python_env = self._detect_pip_environment()
        if self.pip_path:
            print(f"[BoyoAssetGrabber] Detected pip at: {self.pip_path}")
            print(f"[BoyoAssetGrabber] Python environment: {self.python_env}")
            results.append(f"SUCCESS: Detected pip environment - {self.python_env}")
        else:
            print(f"[BoyoAssetGrabber] WARNING: Could not detect pip, requirements.txt installation may fail")
            results.append("WARNING: Could not detect pip environment")
        
        # Process custom nodes
        if "custom_nodes" in manifest:
            print(f"[BoyoAssetGrabber] Processing custom nodes...")
            self.progress_message = "Processing custom nodes..."
            node_results = self._download_custom_nodes(manifest["custom_nodes"], custom_nodes_path)
            results.extend(node_results)
        
        # Process models
        if "models" in manifest:
            print(f"[BoyoAssetGrabber] Processing models...")
            self.progress_message = "Processing models..."
            model_results = self._download_models(manifest["models"], models_path)
            results.extend(model_results)
            
        self.progress_message = ""
        print(f"[BoyoAssetGrabber] Asset processing complete!")
        return "\n".join(results)
    
    def _detect_pip_environment(self) -> Tuple[str, str]:
        """Detect the pip executable and Python environment"""
        # Check if we're in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            env_type = "Virtual Environment"
            venv_path = sys.prefix
        else:
            env_type = "System Python"
            venv_path = sys.executable
            
        # Try different pip locations
        pip_candidates = [
            os.path.join(os.path.dirname(sys.executable), "pip.exe"),  # Windows venv/portable
            os.path.join(os.path.dirname(sys.executable), "pip"),      # Linux venv/portable
            os.path.join(os.path.dirname(sys.executable), "Scripts", "pip.exe"),  # Windows Scripts
            "pip",  # System PATH
            "pip3"  # System PATH alternative
        ]
        
        for pip_path in pip_candidates:
            try:
                # Test if pip works
                result = subprocess.run([pip_path, "--version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return pip_path, f"{env_type} ({venv_path})"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
                
        return None, env_type
    
    def _install_requirements(self, repo_path: str, repo_name: str) -> List[str]:
        """Install requirements.txt if it exists in the cloned repo"""
        results = []
        requirements_path = os.path.join(repo_path, "requirements.txt")
        
        if not os.path.exists(requirements_path):
            print(f"[BoyoAssetGrabber] No requirements.txt found for {repo_name}")
            return results
            
        if not self.pip_path:
            print(f"[BoyoAssetGrabber] ERROR: No pip found, cannot install requirements for {repo_name}")
            results.append(f"ERROR: Cannot install requirements for {repo_name} - pip not found")
            return results
            
        try:
            print(f"[BoyoAssetGrabber] Installing requirements for {repo_name}...")
            self.progress_message = f"Installing requirements for {repo_name}..."
            
            # Run pip install
            result = subprocess.run(
                [self.pip_path, "install", "-r", requirements_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"[BoyoAssetGrabber] Successfully installed requirements for {repo_name}")
                results.append(f"SUCCESS: Installed requirements for {repo_name}")
            else:
                print(f"[BoyoAssetGrabber] ERROR: Failed to install requirements for {repo_name}")
                print(f"[BoyoAssetGrabber] pip error: {result.stderr}")
                results.append(f"ERROR: Failed to install requirements for {repo_name}: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print(f"[BoyoAssetGrabber] ERROR: Timeout installing requirements for {repo_name}")
            results.append(f"ERROR: Timeout installing requirements for {repo_name}")
        except Exception as e:
            print(f"[BoyoAssetGrabber] ERROR: Exception installing requirements for {repo_name}: {str(e)}")
            results.append(f"ERROR: Exception installing requirements for {repo_name}: {str(e)}")
            
        return results
    
    def _download_custom_nodes(self, nodes: List[Dict[str, str]], base_path: str) -> List[str]:
        """Download custom nodes via git clone and install requirements"""
        results = []
        custom_nodes_path = os.path.join(base_path, "custom_nodes")
        
        # Ensure custom_nodes directory exists
        os.makedirs(custom_nodes_path, exist_ok=True)
        print(f"[BoyoAssetGrabber] Custom nodes directory: {custom_nodes_path}")
        
        for i, node in enumerate(nodes, 1):
            repo_url = node["url"]
            repo_name = node["name"]
            target_path = os.path.join(custom_nodes_path, repo_name)
            
            print(f"[BoyoAssetGrabber] Processing custom node {i}/{len(nodes)}: {repo_name}")
            
            if os.path.exists(target_path):
                print(f"[BoyoAssetGrabber] SKIP: {repo_name} already exists at {target_path}")
                results.append(f"SKIP: {repo_name} already exists")
                continue
                
            try:
                print(f"[BoyoAssetGrabber] Cloning {repo_name} from {repo_url}")
                self.progress_message = f"Cloning {repo_name}..."
                git.Repo.clone_from(repo_url, target_path)
                print(f"[BoyoAssetGrabber] Successfully cloned {repo_name}")
                results.append(f"SUCCESS: Cloned {repo_name}")
                
                # Install requirements if they exist
                req_results = self._install_requirements(target_path, repo_name)
                results.extend(req_results)
                
            except Exception as e:
                print(f"[BoyoAssetGrabber] ERROR: Failed to clone {repo_name}: {str(e)}")
                results.append(f"ERROR: Failed to clone {repo_name}: {str(e)}")
                
        return results
    
    def _download_models(self, models: Dict[str, List[Dict]], base_path: str) -> List[str]:
        """Download model files"""
        results = []
        models_base_path = os.path.join(base_path, "models")
        
        print(f"[BoyoAssetGrabber] Models directory: {models_base_path}")
        
        for model_type, model_list in models.items():
            model_dir = os.path.join(models_base_path, model_type)
            os.makedirs(model_dir, exist_ok=True)
            print(f"[BoyoAssetGrabber] Processing {len(model_list)} models for type: {model_type}")
            
            for i, model in enumerate(model_list, 1):
                filename = model["filename"]
                download_url = model["url"]
                target_path = os.path.join(model_dir, filename)
                
                print(f"[BoyoAssetGrabber] Processing model {i}/{len(model_list)}: {filename}")
                
                if os.path.exists(target_path):
                    print(f"[BoyoAssetGrabber] SKIP: {filename} already exists at {target_path}")
                    results.append(f"SKIP: {filename} already exists")
                    continue
                    
                try:
                    print(f"[BoyoAssetGrabber] Downloading {filename} from {download_url}")
                    self.progress_message = f"Downloading {filename}..."
                    self._download_file(download_url, target_path)
                    print(f"[BoyoAssetGrabber] Successfully downloaded {filename}")
                    results.append(f"SUCCESS: Downloaded {filename}")
                except Exception as e:
                    print(f"[BoyoAssetGrabber] ERROR: Failed to download {filename}: {str(e)}")
                    results.append(f"ERROR: Failed to download {filename}: {str(e)}")
                    
        return results
    
    def _download_file(self, url: str, target_path: str) -> None:
        """Download a file from URL to target path"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

NODE_CLASS_MAPPINGS = {
    "BoyoAssetGrabberAdvanced": BoyoAssetGrabberAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoyoAssetGrabberAdvanced": "Boyo Asset Grabber (Advanced)"
}
