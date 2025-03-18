#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from google import genai
import time
from typing import Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("translate-script")


DEFAULT_PROMPT = """You are a LaTeX and {language} language expert. You are tasked with translating lecture notes into {language}.  
Translate the following LaTeX content to {language} by:
1. Translate ONLY the text portions to {language}
2. Leave ALL mathematical content completely unchanged, including:
   - All equation environments and their content
   - All mathematical notation and symbols
   - All variable names and mathematical expressions
3. Preserve all LaTeX environments and structures exactly as they appear
4. DO NOT modify any mathematical expressions or notation
5. DO NOT add any additional content, explanations or commentary
6. Return ONLY the translated LaTeX with mathematics preserved
Input LaTeX follows below:"""


def translate_content(
    input_file_path: str,
    target_language: str = "French",
    output_suffix: str = "translated",
    custom_prompt: Optional[str] = None,
    model: str = "gemini-2.0-flash",
) -> str:
    """
    Translate content from a file to the specified target language,
    preserving special content (like LaTeX math).
    
    Args:
        input_file_path: Path to the input file to translate
        target_language: Target language for translation
        output_suffix: Suffix to add to the output filename
        custom_prompt: Custom prompt to use for translation
        model: Gemini model to use for translation
    
    Returns:
        The path to the translated output file
    """
    input_path = Path(input_file_path)
    
    # Read the input file
    logger.info(f"Reading input file: {input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            content = file.read()
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        sys.exit(1)

    # Initialize Gemini client
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not API_KEY:
        logger.error("GOOGLE_API_KEY environment variable must be set")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)

    # Create the translation prompt
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = DEFAULT_PROMPT.format(language=target_language)

    # Send to Gemini for translation
    logger.info(f"Translating to {target_language} using {model} model...")
    start_time = time.time()
    try:
        response = client.models.generate_content(
            model=model, contents=prompt + "\n\n" + content
        )
        translation_time = time.time() - start_time
        logger.info(f"Translation completed in {translation_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        sys.exit(1)

    # Process the response text - remove markdown code blocks if present
    translated_text = response.text
    
    # Check if the text begins with "```" or "```latex" and ends with "```"
    # This is common in Gemini responses when translating code or LaTeX
    lines = translated_text.splitlines()
    if len(lines) >= 2:
        first_line = lines[0].strip()
        last_line = lines[-1].strip()
        
        # Check if the response is wrapped in code blocks
        if (first_line.startswith("```") and last_line == "```"):
            logger.debug("Detected code block in response, removing wrapper lines")
            # Remove the first and last lines
            translated_text = "\n".join(lines[1:-1])
            logger.debug(f"Removed first line: '{first_line}' and last line: '{last_line}'")

    # Generate output filename
    output_file_path = f"{input_path.stem}_{output_suffix}{input_path.suffix}"
    output_path = input_path.parent / output_file_path
    
    # Save translated content
    logger.info(f"Saving translation to: {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(translated_text)
    except Exception as e:
        logger.error(f"Failed to save translated content: {e}")
        sys.exit(1)

    logger.info(f"Translation complete: {output_path}")
    return str(output_path)


def main():
    """Parse command line arguments and run the translation process"""
    parser = argparse.ArgumentParser(
        description="Translate content from files while preserving special content"
    )
    parser.add_argument(
        "input_file", help="Path to the input file to translate"
    )
    parser.add_argument(
        "-l", "--language", 
        default="French", 
        help="Target language for translation (default: French)"
    )
    parser.add_argument(
        "-o", "--output-suffix", 
        default="translated", 
        help="Suffix to add to the output filename (default: 'translated')"
    )
    parser.add_argument(
        "-p", "--prompt",
        help="Custom prompt to use for translation instead of the default"
    )
    parser.add_argument(
        "-m", "--model",
        default="gemini-2.0-flash",
        help="Gemini model to use for translation (default: gemini-2.0-flash)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Run translation process
    translate_content(
        input_file_path=args.input_file,
        target_language=args.language,
        output_suffix=args.output_suffix,
        custom_prompt=args.prompt,
        model=args.model,
    )


if __name__ == "__main__":
    main()
