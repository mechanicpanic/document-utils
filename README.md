# Document Utils

A collection of utilities for working with documents including translation tools that use Google's Gemini AI model while preserving special content like LaTeX mathematical expressions, markdown formatting, and document structure.

## Features

### Translation Tools
- Supports multiple input formats: PDF, LaTeX, Markdown, and plain text
- Preserves mathematical formulas and special formatting
- Handles large documents through smart batching
- Outputs to multiple formats: TXT, Markdown, JSON, and CSV
- Intelligent line break handling for readable output

### Grammar Checking
- LaTeX grammar checker for improving document quality

## Setup

1. Install the required dependencies:

   Using pip:
   ```
   pip install google-genai>=1.5.0 PyPDF2>=3.0.0 markdown>=3.4.0 PyMuPDF>=1.23.0 rich>=13.0.0
   ```
   
   Or using `uv` (recommended if you have it installed):
   ```
   uv pip install .
   ```

2. Set your Google API key as an environment variable:
   ```
   export GOOGLE_API_KEY="your-api-key-here"
   ```

## Translation Tools

The main script that handles all document types (PDF, LaTeX, Markdown, and plain text):

```
usage: translate.py [-h] [-l LANGUAGE] [-o OUTPUT] [-f FORMATS] [--fix-linebreaks] [-p PROMPT] [-m MODEL] [-b BATCH_SIZE] [-v] input_file

Translate various file formats (PDF, LaTeX, Markdown, Text) while preserving formatting

positional arguments:
  input_file            Path to the input file to translate

options:
  -h, --help            show this help message and exit
  -l LANGUAGE, --language LANGUAGE
                        Target language for translation (default: French)
  -o OUTPUT, --output OUTPUT
                        Base name for output files (default: same as input with _translated suffix)
  -f FORMATS, --formats FORMATS
                        Comma-separated list of output formats: txt,md,json,csv (default: txt,md,json)
  --fix-linebreaks      Fix line breaks in translated output (default: True)
  -p PROMPT, --prompt PROMPT
                        Custom prompt to use for translation instead of the default
  -m MODEL, --model MODEL
                        Gemini model to use for translation (default: gemini-2.0-flash)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Maximum tokens per batch (default: 500000)
  -v, --verbose         Enable verbose logging
```

## Legacy LaTeX Translator

A previously used script specifically for LaTeX translation (kept for backward compatibility):

```
usage: tex_translate.py [-h] [-l LANGUAGE] [-o OUTPUT_SUFFIX] [-p PROMPT] [-m MODEL] [-v] input_file

Translate content from LaTeX files while preserving special content

positional arguments:
  input_file            Path to the input file to translate

options:
  -h, --help            show this help message and exit
  -l LANGUAGE, --language LANGUAGE
                        Target language for translation (default: French)
  -o OUTPUT_SUFFIX, --output-suffix OUTPUT_SUFFIX
                        Suffix to add to the output filename (default: 'translated')
  -p PROMPT, --prompt PROMPT
                        Custom prompt to use for translation instead of the default
  -m MODEL, --model MODEL
                        Gemini model to use for translation (default: gemini-2.0-flash)
  -v, --verbose         Enable verbose logging
```

## Examples

### Translation Examples

Translate a LaTeX file to French (default):
```
./translate.py document.tex
```

Translate a PDF to Spanish and output all format types:
```
./translate.py document.pdf -l Spanish -f txt,md,json,csv
```

Translate a Markdown file using a more powerful model:
```
./translate.py document.md -m gemini-2.0-pro
```

Translate a large document with custom batch size:
```
./translate.py large_document.txt -b 300000
```

### Grammar Checking Examples

Check grammar in a PDF document:
```
./check_grammar.py document.pdf
```

Save grammar check results to a file:
```
./check_grammar.py document.pdf --output grammar_results.json
```

## File Type Detection

The unified script automatically detects the file type based on extension:
- `.pdf` → PDF processing
- `.tex` → LaTeX processing
- `.md`, `.markdown` → Markdown processing
- `.txt`, `.text` → Plain text processing
- Any other extension is treated as plain text

Each file type uses a specialized prompt template to preserve its formatting.

## PDF Slide Translator

A tool for translating PowerPoint presentations converted from PDF files:

```
usage: translate_document_replacement.py [-h] [--output_file OUTPUT_FILE] [--api_key API_KEY] [--skip_conversion]
                                         [--wait_time WAIT_TIME] [--retry_count RETRY_COUNT] [--batch_size BATCH_SIZE] [--save_progress]
                                         input_file target_language

Convert PDF to PPTX and translate using Google Gemini

positional arguments:
  input_file            Path to the input file (PDF or PPTX)
  target_language       Target language for translation

options:
  -h, --help            show this help message and exit
  --output_file         Path to save the translated file
  --api_key             Google API key
  --skip_conversion     Skip PDF to PPTX conversion (if input is already PPTX)
  --wait_time           Wait time between API calls in seconds (default: 1.0)
  --retry_count         Number of retries on API failure (default: 3)
  --batch_size          Number of text blocks to process in each batch (default: 1)
  --save_progress       Save translation progress to resume later if interrupted
```

Features:
- Converts PDF slides to PowerPoint format
- Translates slide text while preserving formatting
- Supports resuming interrupted translations
- Handles rate limiting with exponential backoff

## Grammar Checking Tool

The LaTeX PDF grammar checker extracts text from PDFs and identifies grammar issues:

```
usage: check_grammar.py [-h] [--api-key API_KEY] [--output OUTPUT] pdf_path

LaTeX PDF Grammar Checker

positional arguments:
  pdf_path       Path to the LaTeX PDF file

options:
  -h, --help     show this help message and exit
  --api-key      Google API key for Gemini (or use GOOGLE_API_KEY environment variable)
  --output       Save results to file instead of terminal display
```

Features:
- PDF text extraction with formatting preservation
- Grammar error detection using Gemini AI
- Rich colored terminal output with highlighted differences
- Line-by-line error analysis with suggestions
- Optional JSON output for further processing
