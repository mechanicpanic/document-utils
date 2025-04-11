#!/usr/bin/env python3
"""
LaTeX PDF Grammar Checker
------------------------
This script extracts text from a LaTeX PDF, sends it to Google's Gemini API
for grammar checking, and displays colored terminal output with
line numbers, grammar errors, and suggested fixes with highlighted differences.
"""

import os
import re
import time
import argparse
import json
import difflib
from pathlib import Path
from typing import List, Dict, Tuple, Any

import fitz  # PyMuPDF
from google import genai
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape

# Initialize rich console for colored output
console = Console()


def setup_gemini_client(api_key: str) -> Any:
    """
    Set up and return a Gemini API client.

    Args:
        api_key: Google API key for Gemini

    Returns:
        Configured Gemini client
    """
    # Create a client with the provided API key
    client = genai.Client(api_key=api_key)
    return client


def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """
    Extract text from a PDF file, preserving line breaks.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of lines extracted from the PDF
    """
    console.print(f"[bold blue]Extracting text from[/bold blue] {pdf_path}")

    doc = fitz.open(pdf_path)
    all_lines = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        lines = text.split("\n")
        # Filter out empty lines and page numbers
        lines = [line.strip() for line in lines if line.strip()]
        all_lines.extend(lines)

    console.print(f"[green]Extracted {len(all_lines)} lines of text[/green]")
    return all_lines


def check_grammar_with_gemini(
    client: Any, text_lines: List[str], retry_count: int = 3, wait_time: float = 1.0
) -> Dict:
    """
    Check grammar using the Google Gemini API with rate limiting and retries.

    Args:
        client: Gemini API client
        text_lines: List of text lines from the document
        retry_count: Number of retries on failure
        wait_time: Base wait time between API calls (seconds)

    Returns:
        Dictionary with grammar check results
    """
    # Create a numbered document for better line reference
    numbered_text = ""
    for i, line in enumerate(text_lines):
        numbered_text += f"LINE {i}: {line}\n"

    # Explicitly instruct to return only JSON format with exact line references
    prompt = f"""You are a professional grammar checker for academic writing.
Review the following LaTeX document text for grammar errors only. Each line has a LINE number.

Return your response ONLY in JSON format with the following structure without any explanation:
{{
    "errors": [
        {{
            "line_number": <The exact LINE number from the document>,
            "error_text": <the exact text containing the error>,
            "error_type": <brief description of error type e.g., "Spelling", "Grammar", "Punctuation", etc.>,
            "suggestion": <corrected text>
        }}
    ]
}}

Only include actual grammar issues. Ignore LaTeX commands and formatting.
If there are no errors, return {{"errors": []}}.

IMPORTANT: 
1. Return ONLY valid JSON without any additional text, explanation, or markdown formatting.
2. DO NOT wrap your response in ```json or ``` code blocks.
3. Use EXACTLY the line numbers (LINE X) provided in the document.
4. Include the full corrected sentence or phrase in the suggestion field.

Text to check:
{numbered_text}"""

    attempts = 0
    while attempts < retry_count:
        try:
            # Add wait time before API call to respect rate limits
            time.sleep(wait_time * (1 + attempts))

            # Use the API to generate content
            response = client.models.generate_content(
                model="gemini-2.0-flash-001", contents=prompt
            )

            try:
                # Get just the text from the response
                result = response.text
                console.print(f"[dim]Raw response length: {len(result)} chars[/dim]")

                # Super aggressive JSON cleaning - only keep ASCII printable characters
                json_str = "".join(c for c in result if 32 <= ord(c) <= 126)

                # Remove markdown code blocks if present
                if "```json" in json_str or "```" in json_str:
                    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
                    code_blocks = re.findall(code_block_pattern, json_str)
                    if code_blocks:
                        json_str = code_blocks[0]

                # Find JSON object if not starting with a curly brace
                json_str = json_str.strip()
                if not json_str.startswith("{"):
                    match = re.search(r"({[\s\S]*})", json_str)
                    if match:
                        json_str = match.group(1)

                # Debug log the cleaned JSON
                console.print(f"[dim]Cleaned JSON length: {len(json_str)} chars[/dim]")

                # Last resort: try to fix common JSON issues
                json_str = json_str.replace("\\'", "'").replace('\\"', '"')
                json_str = re.sub(r",\s*}", "}", json_str)  # Remove trailing commas

                try:
                    # Parse the cleaned JSON
                    parsed = json.loads(json_str)

                    # Ensure the response has the expected structure
                    if "errors" not in parsed:
                        parsed = {"errors": []}

                    return parsed
                except json.JSONDecodeError as je:
                    console.print(
                        f"[bold red]Failed to parse JSON response, retrying...[/bold red]"
                    )
                    console.print(f"[dim gray]Error: {str(je)}[/dim gray]")

                    # Try a manual regular expression-based approach to extract errors
                    error_pattern = r'"line_number"\s*:\s*(\d+).*?"error_text"\s*:\s*"([^"]*)".*?"error_type"\s*:\s*"([^"]*)".*?"suggestion"\s*:\s*"([^"]*)"'
                    matches = re.findall(error_pattern, json_str, re.DOTALL)

                    if matches:
                        # Construct a valid JSON manually
                        manual_errors = []
                        for line_num, error_text, error_type, suggestion in matches:
                            manual_errors.append(
                                {
                                    "line_number": int(line_num),
                                    "error_text": error_text,
                                    "error_type": error_type,
                                    "suggestion": suggestion,
                                }
                            )

                        if manual_errors:
                            console.print(
                                f"[yellow]Recovered {len(manual_errors)} errors using regex fallback[/yellow]"
                            )
                            return {"errors": manual_errors}

                    attempts += 1
                    continue

            except Exception as e:
                console.print(f"[bold red]Processing Error: {str(e)}[/bold red]")
                attempts += 1
                continue

        except Exception as e:
            console.print(f"[bold red]API Error: {str(e)}[/bold red]")
            attempts += 1
            if attempts < retry_count:
                console.print(
                    f"[yellow]Retrying in {wait_time * (1 + attempts)} seconds...[/yellow]"
                )

    # If all attempts failed, return an empty result
    console.print(
        f"[bold red]Failed to check grammar after {retry_count} attempts[/bold red]"
    )
    return {"errors": []}


def highlight_differences(original: str, suggestion: str) -> Tuple[Text, Text]:
    """
    Use fuzzy matching to highlight the differences between original and suggested text.

    Args:
        original: Original text with errors
        suggestion: Suggested correction

    Returns:
        Tuple of (formatted_original, formatted_suggestion) with differences highlighted
    """
    # Create Text objects for rich formatting
    original_text = Text()
    suggestion_text = Text()

    # Convert strings to lists of words for comparison
    original_words = original.split()
    suggestion_words = suggestion.split()

    # Use difflib's SequenceMatcher to find differences at the word level
    matcher = difflib.SequenceMatcher(None, original_words, suggestion_words)

    # Process the difflib operations
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Words are the same, add them with normal styling
            for word in original_words[i1:i2]:
                original_text.append(word + " ", style="yellow")
            for word in suggestion_words[j1:j2]:
                suggestion_text.append(word + " ", style="green")

        elif tag == "replace":
            # Words were changed, highlight them
            for word in original_words[i1:i2]:
                original_text.append(word + " ", style="bold red on yellow")
            for word in suggestion_words[j1:j2]:
                suggestion_text.append(word + " ", style="bold black on green")

        elif tag == "delete":
            # Words were deleted from original
            for word in original_words[i1:i2]:
                original_text.append(word + " ", style="bold strike red on yellow")

        elif tag == "insert":
            # Words were added in suggestion
            for word in suggestion_words[j1:j2]:
                suggestion_text.append(
                    word + " ", style="bold underline black on green"
                )

    return original_text, suggestion_text


def display_grammar_results(results: Dict, lines: List[str]) -> None:
    """
    Display grammar checking results with rich formatting and highlighted differences.

    Args:
        results: Grammar check results with errors
        lines: Original text lines
    """
    errors = results.get("errors", [])

    # Sort errors by line number
    errors = sorted(errors, key=lambda x: x.get("line_number", 0))

    if not errors:
        console.print(
            Panel(
                "[bold green]No grammar errors found![/bold green]",
                title="Grammar Check Results",
            )
        )
        return

    console.print(
        Panel(
            f"[bold yellow]Found {len(errors)} grammar issues[/bold yellow]",
            title="Grammar Check Results",
        )
    )

    for error in errors:
        line_num = error.get("line_number", 0)
        error_text = error.get("error_text", "")
        error_type = error.get("error_type", "Unknown error")
        suggestion = error.get("suggestion", "")

        # Ensure we have valid line numbers
        if 0 <= line_num < len(lines):
            # Create rich text display
            error_display = Text()
            error_display.append(f"Line {line_num}: ", style="bold white")
            error_display.append(f"{error_type}\n", style="bold red")

            # Use fuzzy matching to highlight differences
            original_formatted, suggestion_formatted = highlight_differences(
                error_text, suggestion
            )

            error_display.append(f"Original: ", style="bold yellow")
            error_display.append(original_formatted)
            error_display.append("\n")

            error_display.append(f"Suggestion: ", style="bold green")
            error_display.append(suggestion_formatted)

            console.print(Panel(error_display))
        else:
            console.print(f"[yellow]Warning: Invalid line number: {line_num}[/yellow]")


def main():
    """Main function to run the LaTeX PDF grammar checker."""
    parser = argparse.ArgumentParser(description="LaTeX PDF Grammar Checker")
    parser.add_argument("pdf_path", help="Path to the LaTeX PDF file")
    parser.add_argument("--api-key", help="Google API key for Gemini")
    parser.add_argument(
        "--output", help="Save results to file instead of terminal display"
    )
    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        console.print("[bold red]Error: Google API key not provided.[/bold red]")
        console.print(
            "Please provide it using --api-key or set the GOOGLE_API_KEY environment variable."
        )
        return

    # Check if PDF exists
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        console.print(f"[bold red]Error: PDF file not found: {pdf_path}[/bold red]")
        return

    try:
        # Setup client
        client = setup_gemini_client(api_key)

        # Process PDF
        lines = extract_text_from_pdf(str(pdf_path))

        # Process the entire document at once with line numbers included
        with console.status(
            "[bold blue]Checking grammar...[/bold blue]", spinner="dots"
        ) as status:
            results = check_grammar_with_gemini(client, lines)

        # Display results
        display_grammar_results(results, lines)

        # Save results to file if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"[bold green]Results saved to {output_path}[/bold green]")

        console.print("[bold green]Grammar check completed![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Process interrupted by user[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        import traceback

        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()

