# PDF OCR + AI Correction Pipeline

A complete pipeline to convert scanned/handwritten PDFs into searchable documents that work with Samsung Notes and other PDF readers.

## Features

1. **Lossless PDF Compression** - Initial compression with high resolution preservation
2. **PaddleOCR Text Extraction** - Accurate OCR with bounding box detection
3. **VLM-based Correction** - AI-powered OCR error correction using `unsloth/Qwen3-VL-2b-Thinking-bnb-2bit`
4. **Searchable PDF Reconstruction** - Invisible text overlay matching original handwriting positions
5. **Final Optimization** - Deflate compression and garbage collection for minimal file size

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA (if using GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install PaddlePaddle with GPU
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# Or CPU version:
# pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

# Install other dependencies
pip install -r requirements.txt
```

### 3. Set Up VLM Server

The pipeline uses a local VLM server for OCR correction. Choose one option:

#### Option A: Using vLLM (Recommended, requires GPU with 8GB+ VRAM)

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model unsloth/Qwen3-VL-2b-Thinking-bnb-2bit \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 4096
```

#### Option B: Using Ollama (Easier setup)

```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5-vl:2b

# Update config.py:
# VLM_SERVER_URL = "http://localhost:11434/v1"
# VLM_MODEL_NAME = "qwen2.5-vl:2b"
```

#### Option C: Without VLM (OCR only)

```bash
python pipeline.py --no-vlm
```

## Usage

### Process a Single PDF

```bash
python pipeline.py input.pdf -o output.pdf
```

### Process All PDFs in Directory

```bash
# Process all PDFs in 'pdf/' folder, output to 'output/'
python pipeline.py

# Or specify directories
python pipeline.py --input-dir ./my_pdfs --output-dir ./processed
```

### Command Line Options

```
positional arguments:
  input                 Input PDF file

optional arguments:
  -o, --output          Output PDF file
  --input-dir           Input directory for batch processing
  --output-dir          Output directory
  --no-vlm              Skip VLM correction (just use raw OCR)
  --vlm-url             VLM server URL (default: http://localhost:8000/v1)
  --vlm-model           VLM model name
  --no-gpu              Disable GPU for OCR
  --lang                OCR language: en, ch, japan, korean, etc.
```

## Directory Structure

```
sam_pdf/
├── pdf/                    # Input PDFs (place your files here)
│   ├── document.pdf
│   └── subfolder/
│       └── notes.pdf
├── output/                 # Processed PDFs (output here)
│   ├── document.pdf
│   └── subfolder/
│       └── notes.pdf
├── temp/                   # Temporary files (auto-cleaned)
├── config.py               # Configuration settings
├── pipeline.py             # Main pipeline script
├── ocr_extractor.py        # PaddleOCR wrapper
├── vlm_client.py           # VLM client for correction
├── vlm_server.py           # VLM server utilities
├── pdf_compressor.py       # PDF compression utilities
├── pdf_reconstructor.py    # PDF reconstruction with text overlay
└── requirements.txt        # Python dependencies
```

## Configuration

Edit `config.py` to customize:

```python
# OCR Settings
OCR_DPI = 300              # Resolution for OCR (higher = more accurate)
OCR_LANG = "en"            # Language code

# VLM Settings
VLM_SERVER_URL = "http://localhost:8000/v1"
VLM_MODEL_NAME = "unsloth/Qwen3-VL-2b-Thinking-bnb-2bit"

# PDF Output Settings
OUTPUT_DPI = 150           # Output resolution
PDF_GARBAGE_COLLECT = 4    # Compression level (0-4)
TEXT_INVISIBLE = True      # Make OCR text invisible (overlay on handwriting)
```

## How It Works

### 1. Initial Compression

- Loads the PDF and applies lossless deflate compression
- Maintains original image quality for accurate OCR

### 2. OCR Extraction

- Renders each page at 300 DPI
- PaddleOCR detects text regions with precise bounding boxes
- Returns text + coordinates for each detected element

### 3. VLM Correction (Optional)

- Sends page image + OCR results to the VLM
- VLM visually verifies and corrects OCR errors
- Especially useful for handwritten text

### 4. PDF Reconstruction

- Creates new PDF with original visual content
- Adds invisible text layer at exact positions
- Text is searchable but doesn't affect appearance

### 5. Final Compression

- Applies maximum garbage collection (level 4)
- Deflate compression on all streams
- Linearization for faster loading

## Samsung Notes Compatibility

The output PDFs are fully compatible with Samsung Notes:

- Text search works across all pages
- Copy/paste extracted text
- Original handwriting/images preserved
- Optimized file size for mobile devices

## Troubleshooting

### OCR Not Detecting Text

- Increase `OCR_DPI` in config (try 400)
- Ensure PDF images are clear
- Try different language with `--lang`

### VLM Server Connection Failed

- Check server is running: `python vlm_server.py --check`
- Verify URL in config matches server
- Use `--no-vlm` to skip correction

### Output File Too Large

- Reduce `OUTPUT_DPI` in config
- Enable image optimization in compressor
- Check original PDF isn't already compressed

### Missing Dependencies

```bash
# For PaddleOCR issues
pip install paddlepaddle paddleocr --upgrade

# For PyMuPDF issues
pip install PyMuPDF --upgrade
```

## License

MIT License
