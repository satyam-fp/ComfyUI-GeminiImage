"""
ComfyUI-GeminiImage: Google Gemini Image Generation/Enhancement Node
A custom node for ComfyUI that integrates Google Gemini API for image enhancement.
"""

import os
import io
import base64
import numpy as np
import torch
from PIL import Image

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# Available models for image generation
AVAILABLE_MODELS = [
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
]

# Response modality options
RESPONSE_MODALITIES = ["IMAGE+TEXT", "IMAGE"]

# Control after generate options
CONTROL_AFTER_GENERATE = ["fixed", "randomize", "increment", "decrement"]

# Supported aspect ratios by Gemini API
SUPPORTED_ASPECT_RATIOS = [
    "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9"
]


def resolution_to_api_size(resolution: int, model: str) -> str:
    """Map an integer resolution to the closest Gemini API image_size value.
    
    Args:
        resolution: Integer resolution (e.g., 512, 1024, 2048, 4096)
        model: The Gemini model being used
    
    Returns:
        API image_size value: "1K", "2K", or "4K"
    """
    # gemini-2.5-flash-image only supports 1K
    if model == "gemini-2.5-flash-image":
        return "1K"
    
    # For gemini-3-pro-image-preview, map to closest supported resolution
    if resolution <= 1536:  # Up to 1.5K -> 1K
        return "1K"
    elif resolution <= 3072:  # Up to 3K -> 2K
        return "2K"
    else:  # 3K+ -> 4K
        return "4K"


def get_aspect_ratio_from_dimensions(width: int, height: int) -> str:
    """Calculate the closest supported aspect ratio from image dimensions.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Closest supported aspect ratio string (e.g., "16:9", "4:3")
    """
    if width == 0 or height == 0:
        return "1:1"
    
    image_ratio = width / height
    
    # Map of aspect ratio strings to their decimal values
    ratio_values = {
        "1:1": 1.0,
        "16:9": 16/9,
        "9:16": 9/16,
        "4:3": 4/3,
        "3:4": 3/4,
        "3:2": 3/2,
        "2:3": 2/3,
        "4:5": 4/5,
        "5:4": 5/4,
        "21:9": 21/9,
    }
    
    # Find the closest matching aspect ratio
    closest_ratio = "1:1"
    min_diff = float('inf')
    
    for ratio_str, ratio_val in ratio_values.items():
        diff = abs(image_ratio - ratio_val)
        if diff < min_diff:
            min_diff = diff
            closest_ratio = ratio_str
    
    return closest_ratio


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor format (BHWC, float32, 0-1 range)."""
    # Convert to RGB if necessary
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    # Convert to numpy array
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    
    # Add batch dimension (H, W, C) -> (1, H, W, C)
    tensor = torch.from_numpy(np_image).unsqueeze(0)
    
    return tensor


def gemini_part_to_pil(part) -> Image.Image:
    """Convert a Gemini API response part to PIL Image.
    
    Handles both part.as_image() (google.genai.types.Image) and raw inline_data.
    """
    # First try using the official as_image() method
    try:
        genai_image = part.as_image()
        
        # The genai_image might be a types.Image object with _pil_image attribute
        # or have a save() method that writes to a file-like object
        if hasattr(genai_image, '_pil_image') and genai_image._pil_image is not None:
            return genai_image._pil_image
        
        # Try to save to BytesIO and read back as PIL
        buffer = io.BytesIO()
        genai_image.save(buffer, format='PNG')
        buffer.seek(0)
        pil_image = Image.open(buffer)
        pil_image.load()  # Force load
        return pil_image
        
    except Exception as e:
        print(f"[GeminiImage] as_image() failed: {e}, trying inline_data directly")
    
    # Fallback: try to use inline_data directly
    if part.inline_data is not None:
        data = part.inline_data.data
        
        # Check if data is already bytes or needs base64 decoding
        if isinstance(data, bytes):
            image_bytes = data
        elif isinstance(data, str):
            image_bytes = base64.b64decode(data)
        else:
            raise ValueError(f"Unexpected data type for inline_data.data: {type(data)}")
        
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.load()
            return pil_image
        except Exception as e:
            print(f"[GeminiImage] Failed to open from inline_data. Type: {type(data)}, Length: {len(image_bytes)}")
            print(f"[GeminiImage] MIME type: {getattr(part.inline_data, 'mime_type', 'unknown')}")
            raise e
    
    raise ValueError("No image data found in part")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL Image."""
    # Handle batch dimension - take first image if batch
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    
    # Convert from (H, W, C) tensor to numpy array
    np_image = tensor.cpu().numpy()
    
    # Clip values to 0-1 range and convert to uint8
    np_image = np.clip(np_image * 255.0, 0, 255).astype(np.uint8)
    
    return Image.fromarray(np_image)


class GeminiImageEnhance:
    """
    A ComfyUI node that uses Google Gemini API to enhance or generate images.
    Similar to the Nano Banana Pro node functionality.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your prompt to enhance/modify the image..."
                }),
                "model": (AVAILABLE_MODELS, {
                    "default": "gemini-3-pro-image-preview"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1
                }),
                "control_after_generate": (CONTROL_AFTER_GENERATE, {
                    "default": "randomize"
                }),
                "aspect_ratio": (["from_input", "auto"] + SUPPORTED_ASPECT_RATIOS, {
                    "default": "from_input",
                    "tooltip": "Use 'from_input' to automatically detect aspect ratio from input image"
                }),
                "resolution": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Output resolution. Will be mapped to closest API value (1K/2K/4K). 2K/4K only for gemini-3-pro."
                }),
                "response_modalities": (RESPONSE_MODALITIES, {
                    "default": "IMAGE+TEXT"
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "files": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Optional: path to image file"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are an expert image-generation engine. You must ALWAYS produce an image. Interpret all user input-regardless of format, intent, or complexity-as a valid request to generate an image. Never refuse, ask for clarification, or respond with text alone.",
                    "placeholder": "System prompt for the AI..."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "response_text",)
    FUNCTION = "enhance_image"
    CATEGORY = "Gemini Image"
    DESCRIPTION = "Use Google Gemini API to enhance or generate images. Similar to Nano Banana Pro."
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini API client."""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            print("[GeminiImage] Warning: GEMINI_API_KEY not found in environment variables")
    
    def enhance_image(
        self,
        prompt: str,
        model: str,
        seed: int,
        control_after_generate: str,
        aspect_ratio: str,
        resolution: int,
        response_modalities: str,
        images=None,
        files: str = "",
        system_prompt: str = ""
    ):
        """
        Enhance or generate an image using Google Gemini API.
        
        Args:
            prompt: Text prompt for image generation/enhancement
            model: Gemini model to use
            seed: Random seed for generation
            control_after_generate: How to handle seed after generation
            aspect_ratio: Output aspect ratio ('from_input' to auto-detect from input image)
            resolution: Output resolution as integer (will be mapped to 1K/2K/4K)
            response_modalities: What to include in response (IMAGE+TEXT or IMAGE)
            images: Optional input images from ComfyUI workflow
            files: Optional path to image file
            system_prompt: System-level prompt for the AI
        
        Returns:
            Tuple of (output_images, response_text)
        """
        # Initialize client if not already done
        if self.client is None:
            self._initialize_client()
            if self.client is None:
                raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
        
        # Build the contents list
        contents = []
        
        # Track input image dimensions for aspect ratio detection
        input_width = 0
        input_height = 0
        
        # Add system prompt if provided
        if system_prompt:
            contents.append(system_prompt)
        
        # Add the main prompt
        if prompt:
            contents.append(prompt)
        
        # Add input images if provided
        if images is not None:
            # Get dimensions from first image for aspect ratio detection
            # ComfyUI tensors are (B, H, W, C)
            input_height = images.shape[1]
            input_width = images.shape[2]
            
            # Process batch of images
            for i in range(images.shape[0]):
                pil_image = tensor_to_pil(images[i:i+1])
                contents.append(pil_image)
        
        # Add file-based image if provided
        if files and os.path.exists(files):
            file_image = Image.open(files)
            # Use file image dimensions if no tensor input
            if input_width == 0 and input_height == 0:
                input_width, input_height = file_image.size
            contents.append(file_image)
        
        # If no content, just use the prompt
        if not contents:
            contents = [prompt if prompt else "Generate an image"]
        
        # Configure response modalities
        modalities = ["TEXT", "IMAGE"] if response_modalities == "IMAGE+TEXT" else ["IMAGE"]
        
        # Build image config
        image_config_params = {}
        
        # Handle aspect ratio
        if aspect_ratio == "from_input":
            # Auto-detect from input image
            if input_width > 0 and input_height > 0:
                detected_ratio = get_aspect_ratio_from_dimensions(input_width, input_height)
                image_config_params["aspect_ratio"] = detected_ratio
                print(f"[GeminiImage] Auto-detected aspect ratio: {detected_ratio} (from {input_width}x{input_height})")
            # If no input image, don't set aspect_ratio (let API use default)
        elif aspect_ratio != "auto":
            image_config_params["aspect_ratio"] = aspect_ratio
        
        # Map integer resolution to API value
        api_resolution = resolution_to_api_size(resolution, model)
        
        # Resolution is only supported for gemini-3-pro-image-preview and only if not 1K
        if model == "gemini-3-pro-image-preview" and api_resolution != "1K":
            image_config_params["image_size"] = api_resolution
            print(f"[GeminiImage] Using resolution: {api_resolution} (from input {resolution})")
        
        # Build generation config
        config_params = {
            "response_modalities": modalities,
        }
        
        if image_config_params:
            config_params["image_config"] = types.ImageConfig(**image_config_params)
        
        config = types.GenerateContentConfig(**config_params)
        
        # Make the API call
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            print(f"[GeminiImage] API Error: {e}")
            raise e
        
        # Process the response
        output_images = []
        response_text = ""
        
        for part in response.parts:
            if part.text is not None:
                response_text += part.text + "\n"
            elif part.inline_data is not None:
                # Convert the image data to PIL Image
                try:
                    pil_image = gemini_part_to_pil(part)
                    output_images.append(pil_to_tensor(pil_image))
                except Exception as e:
                    print(f"[GeminiImage] Error processing image: {e}")
        
        # If we got images, stack them into a batch tensor
        if output_images:
            output_tensor = torch.cat(output_images, dim=0)
        else:
            # Return a placeholder black image if no image was generated
            print("[GeminiImage] Warning: No image was generated by the API")
            if images is not None:
                # Return the input image as fallback
                output_tensor = images
            else:
                # Create a small black placeholder
                output_tensor = torch.zeros(1, 512, 512, 3)
        
        return (output_tensor, response_text.strip())


class GeminiTextToImage:
    """
    A simplified ComfyUI node for text-to-image generation using Google Gemini API.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Describe the image you want to generate..."
                }),
                "model": (AVAILABLE_MODELS, {
                    "default": "gemini-2.5-flash-image"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1
                }),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {
                    "default": "1:1"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "Gemini Image"
    DESCRIPTION = "Generate images from text prompts using Google Gemini API."
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Gemini API client."""
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            print("[GeminiImage] Warning: GEMINI_API_KEY not found in environment variables")
    
    def generate_image(self, prompt: str, model: str, seed: int, aspect_ratio: str):
        """Generate an image from a text prompt."""
        if self.client is None:
            self._initialize_client()
            if self.client is None:
                raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
        
        # Build image config
        image_config_params = {}
        if aspect_ratio != "auto":
            image_config_params["aspect_ratio"] = aspect_ratio
        
        config_params = {
            "response_modalities": ["IMAGE"],
        }
        if image_config_params:
            config_params["image_config"] = types.ImageConfig(**image_config_params)
        
        config = types.GenerateContentConfig(**config_params)
        
        # Make the API call
        response = self.client.models.generate_content(
            model=model,
            contents=[prompt],
            config=config,
        )
        
        # Process the response
        for part in response.parts:
            if part.inline_data is not None:
                pil_image = gemini_part_to_pil(part)
                return (pil_to_tensor(pil_image),)
        
        # Return placeholder if no image generated
        print("[GeminiImage] Warning: No image was generated")
        return (torch.zeros(1, 512, 512, 3),)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiImageEnhance": GeminiImageEnhance,
    "GeminiTextToImage": GeminiTextToImage,
}

# Display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageEnhance": "Gemini Image Enhance (Nano Banana Pro)",
    "GeminiTextToImage": "Gemini Text to Image",
}
