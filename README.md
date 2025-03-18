# Translate Script

A utility script for translating content using Google's Gemini AI model while preserving special content (such as LaTeX mathematical expressions).

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your Google API key as an environment variable:
   ```
   export GOOGLE_API_KEY="your-api-key-here"
   ```

## Usage

```
usage: translate_gemini.py [-h] [-l LANGUAGE] [-o OUTPUT_SUFFIX] [-p PROMPT] [-m MODEL] [-v] input_file

Translate content from files while preserving special content

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

Translate a LaTeX file to French (default):
```
python translate_gemini.py document.tex
```

Translate a LaTeX file to Spanish with custom output suffix:
```
python translate_gemini.py document.tex --language Spanish --output-suffix ES
```

Use a custom prompt for translation:
```
python translate_gemini.py document.tex --prompt "Translate this to French, focusing on formal academic language"
```

Use a different Gemini model:
```
python translate_gemini.py document.tex --model gemini-2.0-pro
```

## Notes

- The script preserves special content like LaTeX mathematical expressions by instructing the model to translate only text portions.
- The default prompt is designed for LaTeX files but can be customized for other formats.
- Output files are saved in the same directory as the input file with the specified suffix.
- The script automatically removes markdown code block markers (``` and ```) from the beginning and end of the translation response, which Gemini sometimes adds.