# ABBYY FineReader Output

**Note:** ABBYY FineReader is not installed or configured.

## Installation Options

### Linux (ABBYY FineReader Engine)
```bash
# Download from ABBYY website (requires license)
# https://www.abbyy.com/finereader-engine/
```

### Windows (ABBYY FineReader)
```
# Install from: https://www.abbyy.com/finereader/
# Use AbbyyBatchConverter.exe for CLI processing
```

### Command Example
```bash
# Linux:
abbyyocr11 -rl English -if hdf.pdf -of output_abbyy.pdf -f PDF

# Windows:
AbbyyBatchConverter.exe /if "hdf.pdf" /of "output_abbyy.pdf" /f PDF
```

---
*Input file: hdf.pdf*
