import os
import sys
from google import genai
import time


def translate_latex_to_french(input_file_path):
    """Translate a LaTeX file to French preserving all mathematical content"""

    # Read the input file
    with open(input_file_path, "r", encoding="utf-8") as file:
        latex_content = file.read()

    # Initialize Gemini client
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not API_KEY:
        print("Error: GOOGLE_API_KEY environment variable must be set")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)

    # Create the translation prompt
    prompt = """You are a LaTeX and French language expert. You are tasked with translating lecture notes into French.  
Translate the following LaTeX content to French by:
1. Translate ONLY the text portions to French
2. Leave ALL mathematical content completely unchanged, including:
   - All equation environments and their content
   - All mathematical notation and symbols
   - All variable names and mathematical expressions
3. Preserve all LaTeX environments and structures exactly as they appear
4. DO NOT modify any mathematical expressions or notation
5. DO NOT add any additional content, explanations or commentary
6. Return ONLY the translated LaTeX with mathematics preserved
Input LaTeX follows below:"""

    # Send to Gemini for translation
    print(f"Translating {input_file_path}...")
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt + latex_content
    )

    # Generate output filename with _FR suffix
    file_base = os.path.splitext(input_file_path)[0]
    output_file_path = f"{file_base}_translated.tex"

    # Save translated content
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(response.text)

    print(f"Translation saved to {output_file_path}")
    return output_file_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python translate_latex.py input_file.tex")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)

    translate_latex_to_french(input_file)
