#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
import base64
from google import genai
import time
from typing import Optional, List
import json
import fitz  # PyMuPDF
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("handwritten-ocr")

# OCR Prompt for handwritten documents to LaTeX
OCR_PROMPT = """You are an expert OCR system specialized in converting handwritten mathematical and academic documents to LaTeX format.

Analyze the provided handwritten document image and convert it to LaTeX with the following requirements:

1. **Text Recognition**: Accurately recognize all handwritten text and convert it to proper LaTeX
2. **Mathematical Formatting**: 
   - Convert all mathematical expressions to proper LaTeX math mode
   - Use inline math ($...$) for expressions within text
   - Use display math ($$...$$) for standalone equations
   - Properly format fractions, integrals, summations, subscripts, superscripts
3. **Document Structure**: 
   - Preserve the logical structure (sections, subsections, paragraphs)
   - Use appropriate LaTeX commands for headings (\\section{}, \\subsection{}, etc.)
4. **Skip Figures**: Ignore any diagrams, charts, or figures in the document as requested
5. **LaTeX Best Practices**:
   - Use proper LaTeX environments (theorem, proof, definition, etc.) when appropriate
   - Include necessary packages in a comment at the top if special symbols are used
   - Ensure all brackets, braces, and parentheses are properly matched

Return ONLY the LaTeX code without any additional commentary or explanations.

If you cannot read certain parts clearly, use [UNCLEAR] placeholder and continue with the rest of the document.
"""


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    logger.info(f"Encoding image: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        sys.exit(1)


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[str]:
    """
    Convert PDF pages to image files.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for image conversion (default: 300)

    Returns:
        List of paths to the generated image files
    """
    logger.info(f"Converting PDF to images: {pdf_path}")
    image_paths = []

    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)

        # Create output directory for images
        pdf_name = Path(pdf_path).stem
        output_dir = Path(pdf_path).parent / f"{pdf_name}_pages"
        output_dir.mkdir(exist_ok=True)

        logger.info(f"PDF has {len(pdf_document)} pages")

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            # Convert page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scaling factor for DPI
            pix = page.get_pixmap(matrix=mat)

            # Save as PNG
            image_path = output_dir / f"page_{page_num + 1:03d}.png"
            pix.save(str(image_path))
            image_paths.append(str(image_path))

            logger.debug(f"Converted page {page_num + 1} to: {image_path}")

        pdf_document.close()
        logger.info(f"Successfully converted {len(image_paths)} pages to images")

        return image_paths

    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        sys.exit(1)


def get_image_mime_type(image_path: str) -> str:
    """
    Determine the MIME type of an image file.

    Args:
        image_path: Path to the image file

    Returns:
        MIME type string
    """
    extension = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }
    return mime_types.get(extension, "image/jpeg")


def process_handwritten_document(
    image_path: str,
    custom_prompt: Optional[str] = None,
    model: str = "gemini-2.0-flash",
) -> str:
    """
    Process handwritten document image and convert to LaTeX using Gemini.

    Args:
        image_path: Path to the input image file
        custom_prompt: Custom prompt to use instead of default
        model: Gemini model to use for OCR

    Returns:
        LaTeX formatted text from the handwritten document
    """
    # Initialize Gemini client
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not API_KEY:
        logger.error("GOOGLE_API_KEY environment variable must be set")
        sys.exit(1)

    client = genai.Client(api_key=API_KEY)

    # Use custom prompt if provided, otherwise use default
    prompt = custom_prompt if custom_prompt else OCR_PROMPT

    # Encode the image
    base64_image = encode_image_to_base64(image_path)

    logger.info(f"Processing handwritten document with {model} model...")
    start_time = time.time()

    try:
        # Create the content with both text and image
        content = [
            prompt,
            {
                "inline_data": {
                    "mime_type": get_image_mime_type(image_path), 
                    "data": base64_image
                }
            },
        ]

        # Create generation config for better OCR results
        generation_config = genai.types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=0.1,  # Lower temperature for more accurate OCR
        )

        response = client.models.generate_content(
            model=model, contents=content, config=generation_config
        )

        processing_time = time.time() - start_time
        logger.info(f"OCR processing completed in {processing_time:.2f} seconds")

        if not response.text:
            logger.error("No text was generated from the image")
            sys.exit(1)

        return response.text

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        sys.exit(1)


def process_multiple_images(
    image_paths: List[str],
    custom_prompt: Optional[str] = None,
    model: str = "gemini-2.0-flash",
) -> str:
    """
    Process multiple handwritten document images and combine LaTeX output.

    Args:
        image_paths: List of paths to image files
        custom_prompt: Custom prompt to use instead of default
        model: Gemini model to use for OCR

    Returns:
        Combined LaTeX formatted text from all images
    """
    logger.info(f"Processing {len(image_paths)} images...")
    latex_parts = []

    for i, image_path in enumerate(image_paths):
        logger.info(
            f"Processing image {i + 1}/{len(image_paths)}: {Path(image_path).name}"
        )

        # Add page break comment for multi-page documents
        if i > 0:
            latex_parts.append("\n% Page break\n\\newpage\n")

        latex_content = process_handwritten_document(
            image_path=image_path, custom_prompt=custom_prompt, model=model
        )

        latex_parts.append(latex_content)

    # Combine all parts
    combined_latex = "\n".join(latex_parts)
    logger.info(f"Successfully processed all {len(image_paths)} images")

    return combined_latex


def save_latex_output(
    latex_content: str, output_path: str, include_preamble: bool = True
) -> str:
    """
    Save the LaTeX content to a file with optional document preamble.

    Args:
        latex_content: The LaTeX content to save
        output_path: Path for the output file
        include_preamble: Whether to include a basic LaTeX document preamble

    Returns:
        Path to the saved file
    """
    logger.info(f"Saving LaTeX output to: {output_path}")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            if include_preamble:
                # Add basic LaTeX document structure
                f.write("\\documentclass{article}\n")
                f.write("\\usepackage[utf8]{inputenc}\n")
                f.write("\\usepackage{amsmath}\n")
                f.write("\\usepackage{amssymb}\n")
                f.write("\\usepackage{amsfonts}\n")
                f.write("\\usepackage{theorem}\n")
                f.write("\n")
                f.write("\\begin{document}\n")
                f.write("\n")
                f.write(latex_content)
                f.write("\n")
                f.write("\\end{document}\n")
            else:
                f.write(latex_content)

        logger.info(f"LaTeX content saved successfully")
        return output_path

    except Exception as e:
        logger.error(f"Failed to save LaTeX output: {e}")
        sys.exit(1)


def save_metadata(
    source_path: str,
    output_path: str,
    model: str,
    processing_time: float,
    file_type: str,
    num_pages: int = 1,
) -> str:
    """
    Save metadata about the OCR process to a JSON file.

    Args:
        source_path: Path to the original file (image or PDF)
        output_path: Path to the LaTeX output file
        model: Model used for OCR
        processing_time: Time taken for processing
        file_type: Type of input file ('image' or 'pdf')
        num_pages: Number of pages processed

    Returns:
        Path to the metadata file
    """
    metadata_path = output_path.replace(".tex", "_metadata.json")

    metadata = {
        "source_file": source_path,
        "file_type": file_type,
        "num_pages": num_pages,
        "output_file": output_path,
        "model_used": model,
        "processing_time_seconds": processing_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_version": "2.0",
    }

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Metadata saved to: {metadata_path}")
        return metadata_path

    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        return ""


def validate_input_file(file_path: str) -> str:
    """
    Validate that the input file is a supported format and return the file type.

    Args:
        file_path: Path to the input file

    Returns:
        File type ('image' or 'pdf')
    """
    supported_image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
    }
    supported_pdf_extensions = {".pdf"}

    file_extension = Path(file_path).suffix.lower()

    if file_extension in supported_image_extensions:
        return "image"
    elif file_extension in supported_pdf_extensions:
        return "pdf"
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        logger.error(
            f"Supported formats: {', '.join(supported_image_extensions | supported_pdf_extensions)}"
        )
        sys.exit(1)


def main():
    """Parse command line arguments and run the OCR process"""
    parser = argparse.ArgumentParser(
        description="Convert handwritten documents to LaTeX using Gemini OCR"
    )
    parser.add_argument(
        "input_file", help="Path to the handwritten document image file or PDF"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output LaTeX file path (default: same as input with .tex extension)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF to image conversion (default: 300)",
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep intermediate image files when processing PDFs",
    )
    parser.add_argument(
        "-p", "--prompt", help="Custom prompt to use for OCR instead of the default"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model to use for OCR (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--no-preamble",
        action="store_true",
        help="Don't include LaTeX document preamble (output content only)",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save processing metadata to a JSON file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Validate input file format and get type
    file_type = validate_input_file(str(input_path))

    # Set output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path.with_suffix(".tex"))

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process the document based on file type
    start_time = time.time()

    if file_type == "pdf":
        # Convert PDF to images first
        logger.info("Converting PDF to images for OCR processing...")
        image_paths = pdf_to_images(str(input_path), dpi=args.dpi)

        # Process all images
        latex_content = process_multiple_images(
            image_paths=image_paths,
            custom_prompt=args.prompt,
            model=args.model,
        )

        # Clean up intermediate images unless requested to keep them
        if not args.keep_images:
            logger.info("Cleaning up intermediate image files...")
            images_dir = Path(image_paths[0]).parent
            try:
                for img_path in image_paths:
                    Path(img_path).unlink()
                images_dir.rmdir()
                logger.debug("Intermediate images cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up intermediate images: {e}")

        num_pages = len(image_paths)
    else:
        # Process single image
        latex_content = process_handwritten_document(
            image_path=str(input_path),
            custom_prompt=args.prompt,
            model=args.model,
        )
        num_pages = 1

    total_time = time.time() - start_time

    # Save the LaTeX output
    saved_file = save_latex_output(
        latex_content=latex_content,
        output_path=output_path,
        include_preamble=not args.no_preamble,
    )

    # Save metadata if requested
    if args.save_metadata:
        save_metadata(
            source_path=str(input_path),
            output_path=output_path,
            model=args.model,
            processing_time=total_time,
            file_type=file_type,
            num_pages=num_pages,
        )

    logger.info(f"OCR processing complete. LaTeX saved to: {saved_file}")
    logger.info(f"Processed {num_pages} page(s) in {total_time:.2f} seconds")
    if file_type == "pdf":
        logger.info(f"Average time per page: {total_time / num_pages:.2f} seconds")


if __name__ == "__main__":
    main()

