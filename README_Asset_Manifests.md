# Asset Manifest Creation Guide

This guide explains how to create JSON manifest files for the Boyo Asset Grabber system. These manifests define all the custom nodes and models needed for a ComfyUI workflow, enabling one-click installation for end users.

## Overview

An asset manifest is a JSON file that describes:
- **Custom nodes** to clone from GitHub repositories
- **Models** to download from direct URLs
- **System requirements** and compatibility info
- **Workflow metadata** for categorisation

## Quick Start

1. Copy the example manifest below
2. Update the `name`, `description`, and `version`
3. Add your custom nodes to the `custom_nodes` array
4. Add your models to the `models` object, organised by type
5. Validate your JSON using the schema
6. Test with the Asset Grabber nodes

## Schema Validation

Use the included `asset_manifest_schema.json` to validate your manifests:
- **Online**: Use [JSONSchemaLint](https://jsonschemalint.com/)
- **VS Code**: Install JSON Schema extension for real-time validation
- **Command line**: Use tools like `ajv-cli`

## Manifest Structure

### Required Fields

```json
{
  "name": "Your Workflow Name",
  "version": "1.0.0"
}
```

### Basic Example

```json
{
  "name": "Audio Video Workflow",
  "description": "Complete SFW/NSFW audio generation for videos",
  "version": "1.0.0",
  "author": "Boyo",
  "custom_nodes": [
    {
      "name": "ComfyUI-HunyuanVideo-Foley",
      "url": "https://github.com/phazei/ComfyUI-HunyuanVideo-Foley",
      "description": "Foley sound generation for videos"
    }
  ],
  "models": {
    "foley": [
      {
        "filename": "hunyuan_foley_model.safetensors",
        "url": "https://huggingface.co/user/repo/resolve/main/model.safetensors",
        "description": "Main foley generation model",
        "size_gb": 5.2
      }
    ]
  }
}
```

## Custom Nodes Section

Add each required custom node repository:

```json
"custom_nodes": [
  {
    "name": "ComfyUI-NodePackName",           // Folder name (repository name)
    "url": "https://github.com/user/repo",   // GitHub repository URL
    "description": "Optional description",    // What this node pack does
    "branch": "main"                         // Optional: specific branch
  }
]
```

### Custom Node Guidelines

- **Name**: Must match the GitHub repository name exactly
- **URL**: Must be a valid GitHub repository URL
- **Requirements**: The system automatically installs `requirements.txt` if present
- **Branch**: Defaults to main/master if not specified

## Models Section

Organise models by logical categories:

```json
"models": {
  "checkpoints": [
    {
      "filename": "my_model.safetensors",
      "url": "https://direct-download-url.com/model.safetensors",
      "description": "Main generation model",
      "size_gb": 4.2,
      "required": true
    }
  ],
  "loras": [
    {
      "filename": "style_lora.safetensors", 
      "url": "https://another-url.com/lora.safetensors",
      "size_mb": 150,
      "required": false
    }
  ],
  "embeddings": [
    {
      "filename": "negative_embedding.pt",
      "url": "https://embedding-url.com/file.pt"
    }
  ]
}
```

### Model Categories

Common categories (create your own as needed):
- `checkpoints` - Main diffusion models
- `loras` - LoRA adapters  
- `embeddings` - Textual inversions
- `upscale_models` - Super resolution models
- `controlnet` - ControlNet models
- `vae` - Variational autoencoders
- `clip` - CLIP models
- `foley` - Audio generation models
- `mmaudio` - Multi-modal audio models

### Model URLs

**Direct Download Requirements:**
- Must be HTTPS URLs
- Must return the actual file (not a webpage)
- No authentication required
- No redirects through download pages

**Good URL Examples:**
```
https://huggingface.co/user/repo/resolve/main/model.safetensors
https://civitai.com/api/download/models/12345
https://github.com/user/repo/releases/download/v1.0/model.safetensors
```

**Bad URL Examples:**
```
https://huggingface.co/user/repo/blob/main/model.safetensors  // blob, not resolve
https://civitai.com/models/12345                             // model page, not download
https://drive.google.com/file/d/...                          // requires auth
```

### File Size Guidelines

Include file sizes for better user experience:
```json
{
  "filename": "large_model.safetensors",
  "url": "https://...",
  "size_gb": 5.2,     // For files >= 1GB
  "size_mb": 850      // For files < 1GB
}
```

## Requirements Section

Specify system requirements and compatibility:

```json
"requirements": {
  "disk_space_gb": 12.5,
  "vram_gb": 8,
  "python_version": "3.8+",
  "comfyui_version": ">=1.0.0",
  "notes": [
    "Requires GPU with at least 8GB VRAM for optimal performance",
    "NSFW content generation available with MMAudio models",
    "Some models require additional setup - see documentation"
  ]
}
```

## Workflow Info Section

Add metadata about your workflow:

```json
"workflow_info": {
  "workflow_file": "audio_video_generation.json",
  "input_types": ["video", "text"],
  "output_types": ["video", "audio"],
  "tags": ["audio", "video", "nsfw", "generation"]
}
```

## Best Practices

### File Organisation

1. **Use clear model categories** - Group related models logically
2. **Descriptive filenames** - Include version, type, or variant in names
3. **Consistent naming** - Use same naming convention across workflows

### URL Management

1. **Fork repositories** - Fork models to your own HuggingFace/GitHub for stability
2. **Rename models** - Give models descriptive names instead of `pytorch_model.safetensors`
3. **Version control** - Include version numbers in URLs when possible
4. **Test all URLs** - Verify every URL downloads correctly before publishing

### User Experience

1. **Size information** - Always include file sizes for large models
2. **Clear descriptions** - Explain what each component does
3. **Requirements** - Be specific about hardware/software needs
4. **Notes** - Include any setup steps or warnings

### Testing

Before distributing your manifest:

1. **Clean install test** - Test on fresh ComfyUI installation
2. **Validate JSON** - Use schema validation
3. **Check all URLs** - Verify every download link works
4. **Test workflow** - Ensure the workflow runs after asset installation

## Troubleshooting

### Common Issues

**"No JSON files found"**
- Check file is in `custom_nodes/Boyonodes/assetJsons/`
- Ensure file has `.json` extension
- Restart ComfyUI after adding files

**"ERROR: JSON file not found"**
- Verify JSON syntax is valid
- Check file permissions
- Ensure no hidden characters in filename

**Download failures**
- Test URLs manually in browser
- Check for redirects or auth requirements
- Verify URLs return actual files, not HTML pages

**Git clone failures**
- Ensure repository URLs are accessible
- Check for typos in repository names
- Verify repositories exist and are public

## Example Complete Manifest

```json
{
  "name": "Character Generation Workflow",
  "description": "Complete character generation with multiple styles and poses",
  "version": "2.1.0", 
  "author": "Dragon Diffusion",
  "custom_nodes": [
    {
      "name": "ComfyUI-Advanced-ControlNet",
      "url": "https://github.com/Fannovel16/ComfyUI-Advanced-ControlNet",
      "description": "Advanced ControlNet implementations"
    },
    {
      "name": "ComfyUI-Impact-Pack", 
      "url": "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
      "description": "Detection and segmentation tools"
    }
  ],
  "models": {
    "checkpoints": [
      {
        "filename": "character_base_v2.safetensors",
        "url": "https://huggingface.co/dragonDiffusion/CharacterPack/resolve/main/character_base_v2.safetensors",
        "description": "Main character generation model",
        "size_gb": 4.2,
        "sha256": "a1b2c3d4e5f6..."
      }
    ],
    "loras": [
      {
        "filename": "pose_control_v1.safetensors",
        "url": "https://huggingface.co/dragonDiffusion/CharacterPack/resolve/main/pose_control_v1.safetensors", 
        "description": "Pose and gesture control LoRA",
        "size_mb": 150
      }
    ],
    "controlnet": [
      {
        "filename": "openpose_character.safetensors",
        "url": "https://huggingface.co/dragonDiffusion/CharacterPack/resolve/main/openpose_character.safetensors",
        "description": "OpenPose model optimised for character generation", 
        "size_gb": 1.4
      }
    ]
  },
  "requirements": {
    "disk_space_gb": 8.5,
    "vram_gb": 6,
    "python_version": "3.8+",
    "comfyui_version": ">=1.0.0",
    "notes": [
      "Optimised for character generation and portraits",
      "Includes multiple pose control options",
      "Compatible with standard ControlNet workflows"
    ]
  },
  "workflow_info": {
    "workflow_file": "character_generation_v2.json",
    "input_types": ["text", "image"],
    "output_types": ["image"],
    "tags": ["character", "portrait", "controlnet", "generation"]
  }
}
```

## Distribution

Once your manifest is ready:

1. **Save** as `.json` file with descriptive name
2. **Validate** using the schema  
3. **Test** with Asset Grabber nodes
4. **Distribute** to customers via Patreon/website
5. **Include** installation instructions

Your customers simply need to:
1. Download the JSON file
2. Drop it in `custom_nodes/Boyonodes/assetJsons/`
3. Restart ComfyUI
4. Select from dropdown and click download

## Support

For issues with the Asset Grabber system:
- Check console output for detailed error messages
- Verify all URLs are accessible
- Ensure JSON syntax is valid
- Test on clean ComfyUI installation

The Asset Grabber provides detailed logging in the console window to help diagnose any issues during the download process.
