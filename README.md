# ComfyUI-GeminiImage

A custom ComfyUI node package that integrates **Google Gemini API** for AI-powered image generation and enhancement.

![Category](https://img.shields.io/badge/Category-Gemini%20Image-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)

---

## ✨ Features

- **Gemini Image Enhance** - Enhance, edit, or transform existing images using text prompts
- **Gemini Text to Image** - Generate images from text descriptions
- **Multiple Models** - Support for `gemini-3-pro-image-preview` and `gemini-2.5-flash-image`
- **Flexible Aspect Ratios** - Auto, 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 4:5, 5:4, 21:9
- **Resolution Options** - 1K, 2K, and 4K output (2K/4K requires `gemini-3-pro-image-preview`)
- **Batch Processing** - Process multiple images in a single workflow

---

## 📦 Installation

### Option 1: Clone via Git (Recommended)

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/satyam-fp/ComfyUI-GeminiImage.git
cd ComfyUI-GeminiImage
pip install -r requirements.txt
```

### Option 2: Manual Installation

1. Download or clone this repository
2. Place the `ComfyUI-GeminiImage` folder in `ComfyUI/custom_nodes/`
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Version |
|---------|---------|
| google-genai | ≥1.0.0 |
| Pillow | ≥9.0.0 |
| python-dotenv | ≥1.0.0 |

---

## 🔑 Configuration

### Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Navigate to **Get API Key** → **Create API Key**
4. Copy your API key

### Setting Up Your API Key

Create a `.env` file in your ComfyUI root directory (or the directory where you run ComfyUI):

```bash
# .env
GEMINI_API_KEY=your_api_key_here
```

> [!IMPORTANT]
> Never commit your `.env` file to version control. Add `.env` to your `.gitignore`.

---

## 🎨 Available Nodes

### 1. Gemini Image Enhance

**Category:** `Gemini Image`
**Node Name:** `Gemini Image Enhance (Nano Banana Pro)`

Multi-purpose node for image enhancement, editing, and generation.

#### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `prompt` | STRING | ✅ | Text prompt to enhance/modify the image |
| `model` | COMBO | ✅ | Gemini model to use |
| `seed` | INT | ✅ | Random seed for generation |
| `control_after_generate` | COMBO | ✅ | Seed behavior: fixed, randomize, increment, decrement |
| `aspect_ratio` | COMBO | ✅ | Output aspect ratio |
| `resolution` | COMBO | ✅ | Output resolution (1K, 2K, 4K) |
| `response_modalities` | COMBO | ✅ | IMAGE+TEXT or IMAGE |
| `images` | IMAGE | ❌ | Optional input images from workflow |
| `files` | STRING | ❌ | Optional path to image file |
| `system_prompt` | STRING | ❌ | System-level prompt for the AI |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `images` | IMAGE | Generated/enhanced image(s) |
| `response_text` | STRING | Text response from the AI (if IMAGE+TEXT) |

---

### 2. Gemini Text to Image

**Category:** `Gemini Image`
**Node Name:** `Gemini Text to Image`

Simplified text-to-image generation node.

#### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `prompt` | STRING | ✅ | Text description of the image |
| `model` | COMBO | ✅ | Gemini model to use |
| `seed` | INT | ✅ | Random seed for generation |
| `aspect_ratio` | COMBO | ✅ | Output aspect ratio |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Generated image |

---

## 🚀 Quick Start

### Text to Image Example

1. Add a **Gemini Text to Image** node
2. Enter your prompt: *"A majestic mountain landscape at sunset with golden clouds"*
3. Select model: `gemini-2.5-flash-image`
4. Choose aspect ratio: `16:9`
5. Connect to a **Preview Image** or **Save Image** node
6. Run the workflow

### Image Enhancement Example

1. Load an image using **Load Image** node
2. Add a **Gemini Image Enhance** node
3. Connect the image to the `images` input
4. Enter enhancement prompt: *"Add more dramatic lighting and enhance colors"*
5. Connect to output and run

---

## 📌 Model Comparison

| Model | Speed | Quality | 2K/4K Support | Best For |
|-------|-------|---------|---------------|----------|
| `gemini-2.5-flash-image` | ⚡ Fast | Good | ❌ | Quick iterations, drafts |
| `gemini-3-pro-image-preview` | 🐢 Slower | Excellent | ✅ | Final renders, high quality |

---

## 🔧 Troubleshooting

### "GEMINI_API_KEY not found" Error

- Ensure your `.env` file is in the correct location
- Verify the API key is correctly formatted without quotes (unless they're part of the key)
- Restart ComfyUI after creating/modifying the `.env` file

### No Image Generated

- Check your Gemini API quota at [Google AI Studio](https://aistudio.google.com/)
- Ensure your prompt doesn't violate content policies
- Try a different model

### Import Errors

Verify all dependencies are installed:

```bash
pip install google-genai Pillow python-dotenv
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📬 Support

- **Issues:** [GitHub Issues](https://github.com/satyam-fp/ComfyUI-GeminiImage/issues)
- **Repository:** [ComfyUI-GeminiImage](https://github.com/satyam-fp/ComfyUI-GeminiImage)

---

## 🙏 Acknowledgments

- [Google Gemini API](https://ai.google.dev/) for the powerful AI image generation
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the amazing node-based workflow system
