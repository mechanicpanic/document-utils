#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
import tempfile
from google import genai
import time
from typing import Optional, List, Tuple
import PyPDF2
import markdown
import json
import csv
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("translate-unified")

# Constants
MAX_TOKENS = 2000  # Maximum tokens per batch - conservative to ensure complete outputs from each API call

# Prompt templates for different file types
LATEX_PROMPT = """You are a LaTeX and {language} language expert. You are tasked with translating lecture notes into {language}.  
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

PDF_PROMPT = """You are a professional translator specializing in {language}. 
Translate the following PDF text into {language}:
1. Maintain paragraph structure and formatting when possible
2. Preserve names, dates, and proper nouns as appropriate
3. Translate technical terms accurately using domain-appropriate language
4. Format all mathematical formulas and equations as proper LaTeX with dollar signs:
   - Inline formulas should use single dollar signs ($...$)
   - Display/block equations should use double dollar signs ($$...$$)
   - Do NOT translate any content inside LaTeX formulas
5. Do not add extra line breaks or newlines - maintain normal paragraph flow
6. Return only the translated text without additional commentary

Text to translate:"""

CRITICAL_THEORY_PROMPT = """You are a highly specialized academic translator with expertise in critical theory, philosophy, and social sciences. 
Translate the following PhD thesis on critical theory into {language} with precision:

1. Maintain the academic register and tone throughout the translation
2. Preserve all specialized terminology from critical theory, philosophy, and social sciences
3. Pay careful attention to philosophical concepts and maintain consistent translations for key terms
4. Correctly translate citations and references according to academic conventions in the target language
5. Preserve the logical structure and argumentation flow of the original text
6. Maintain paragraph structure, section headings, and formatting
7. Ensure footnotes, endnotes, and bibliographic entries are correctly formatted in the target language
8. Translate quoted material appropriately while indicating it remains a quotation
9. Preserve proper names, institution names, and titles of works as appropriate in the target language
10. Be attentive to subtle distinctions in theoretical language and maintain philosophical nuance
11. Return only the translated text without additional commentary. DO NOT ADD any extra information or context.

PhD Thesis text to translate:"""

MARKDOWN_PROMPT = """You are a markdown and {language} language expert. 
Translate the following markdown content to {language} by:
1. Translate ONLY the text portions to {language}
2. Preserve all markdown formatting (headers, lists, links, etc.)
3. Leave code blocks and inline code untranslated
4. Format all mathematical formulas correctly with dollar signs:
   - Inline formulas should use single dollar signs ($...$)
   - Display/block equations should use double dollar signs ($$...$$)
5. Preserve all links, images, and other markdown elements
6. Maintain document structure and formatting
7. Return ONLY the translated markdown

Input markdown follows below:"""

TEXT_PROMPT = """You are a professional translator specializing in {language}. 
Translate the following text into {language}:
1. Maintain paragraph structure and formatting when possible
2. Preserve names, dates, and proper nouns as appropriate
3. Translate technical terms accurately using domain-appropriate language
4. Do not add extra line breaks or newlines - maintain normal paragraph flow
5. Return only the translated text without additional commentary

Text to translate:"""


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text from the PDF
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    try:
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            logger.info(f"PDF has {num_pages} pages")

            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()

                if page_text:
                    # PDF extraction often has newlines after every word
                    # Fix this by replacing lone newlines, but keep paragraph breaks
                    processed_text = re.sub(r"(?<!\n)\n(?!\n)", " ", page_text)
                    # Remove multiple spaces
                    processed_text = re.sub(r" +", " ", processed_text)
                    # Ensure consistent paragraph breaks
                    processed_text = re.sub(r"\n{3,}", "\n\n", processed_text)

                    text += f"--- Page {page_num + 1} ---\n{processed_text}\n\n"
                else:
                    logger.warning(f"No text found on page {page_num + 1}")

        if not text.strip():
            logger.warning("No text was extracted from the PDF")

        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        sys.exit(1)


def read_latex_file(file_path: str) -> str:
    """
    Read LaTeX content from a file.

    Args:
        file_path: Path to the LaTeX file

    Returns:
        LaTeX content as string
    """
    logger.info(f"Reading LaTeX file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        logger.error(f"Failed to read LaTeX file: {e}")
        sys.exit(1)


def read_markdown_file(file_path: str) -> str:
    """
    Read Markdown content from a file.

    Args:
        file_path: Path to the Markdown file

    Returns:
        Markdown content as string
    """
    logger.info(f"Reading Markdown file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        logger.error(f"Failed to read Markdown file: {e}")
        sys.exit(1)


def read_text_file(file_path: str) -> str:
    """
    Read plain text content from a file.

    Args:
        file_path: Path to the text file

    Returns:
        Text content as string
    """
    logger.info(f"Reading text file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        logger.error(f"Failed to read text file: {e}")
        sys.exit(1)


def detect_file_type(file_path: str) -> Tuple[str, str]:
    """
    Detect file type based on extension and return the appropriate prompt template.

    Args:
        file_path: Path to the input file

    Returns:
        Tuple of (file_type, prompt_template)
    """
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        # return "pdf", PDF_PROMPT
        return "pdf", CRITICAL_THEORY_PROMPT
    elif extension == ".tex":
        return "latex", LATEX_PROMPT
    elif extension in [".md", ".markdown"]:
        return "markdown", MARKDOWN_PROMPT
    elif extension in [".txt", ".text"]:
        return "text", TEXT_PROMPT
    else:
        logger.warning(f"Unknown file extension: {extension}. Treating as plain text.")
        return "text", TEXT_PROMPT


def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in the text.
    This is a rough approximation: ~4 characters per token for English.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated number of tokens
    """
    # Simple approximation: ~4 characters per token for English
    return len(text) // 4


def split_text_into_batches(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """
    Split the text into batches that fit within token limits.
    Tries to split at page boundaries or paragraphs.

    Args:
        text: The text to split
        max_tokens: Maximum tokens per batch

    Returns:
        List of text batches
    """
    estimated_tokens = estimate_token_count(text)

    if estimated_tokens <= max_tokens:
        return [text]

    batches = []
    # Look for page markers to split logically
    page_splits = re.split(r"(--- Page \d+ ---)", text)

    current_batch = ""
    current_tokens = 0

    # If we have page markers, try to split by pages
    if len(page_splits) > 1:
        for i, part in enumerate(page_splits):
            part_tokens = estimate_token_count(part)

            # If adding this part would exceed the limit, start a new batch
            if current_tokens + part_tokens > max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = part
                current_tokens = part_tokens
            else:
                current_batch += part
                current_tokens += part_tokens
    else:
        # Fall back to paragraph splitting if no page markers
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para_tokens = estimate_token_count(para)

            # If a single paragraph exceeds limit, we'll have to split it
            if para_tokens > max_tokens:
                # Split the paragraph into sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sentence_batch = ""
                sentence_tokens = 0

                for sentence in sentences:
                    sentence_tokens = estimate_token_count(sentence)
                    if (
                        sentence_tokens + sentence_tokens > max_tokens
                        and sentence_batch
                    ):
                        batches.append(sentence_batch)
                        sentence_batch = sentence + " "
                        sentence_tokens = sentence_tokens
                    else:
                        sentence_batch += sentence + " "
                        sentence_tokens += sentence_tokens

                if sentence_batch:
                    if current_batch and (
                        current_tokens + sentence_tokens <= max_tokens
                    ):
                        current_batch += "\n\n" + sentence_batch
                        current_tokens += sentence_tokens
                    else:
                        if current_batch:
                            batches.append(current_batch)
                        current_batch = sentence_batch
                        current_tokens = sentence_tokens
            elif current_tokens + para_tokens > max_tokens:
                batches.append(current_batch)
                current_batch = para
                current_tokens = para_tokens
            else:
                if current_batch:
                    current_batch += "\n\n" + para
                else:
                    current_batch = para
                current_tokens += para_tokens

    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)

    logger.info(f"Split text into {len(batches)} batches")
    return batches


def translate_content(
    content: str,
    prompt_template: str,
    target_language: str = "French",
    custom_prompt: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    batch_size: int = MAX_TOKENS,
    use_streaming: bool = False,
) -> str:
    """
    Translate content to the specified target language using Gemini.
    Handles large content by splitting into batches if needed.

    Args:
        content: The text content to translate
        prompt_template: The template to use for translation prompts
        target_language: Target language for translation
        custom_prompt: Custom prompt to use for translation
        model: Gemini model to use for translation
        batch_size: Maximum batch size in tokens

    Returns:
        The translated text
    """
    # Initialize Gemini client
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not API_KEY:
        logger.error("GOOGLE_API_KEY environment variable must be set")
        sys.exit(1)

    # Initialize client with a more complete configuration
    client = genai.Client(
        api_key=API_KEY,
    )

    # Log client configuration
    logger.debug(f"Client initialized with transport: rest, timeout: 300s")

    # Create the translation prompt
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = prompt_template.format(language=target_language)

    # Check if we need to batch the content
    batches = split_text_into_batches(content, max_tokens=batch_size)
    logger.debug(f"Input content length: {len(content)} chars")

    if len(batches) == 1:
        # Single batch translation
        logger.info(
            f"Translating entire text to {target_language} using {model} model..."
        )
        start_time = time.time()
        try:
            full_prompt = prompt + "\n\n" + content
            logger.debug(f"Full prompt length: {len(full_prompt)} chars")

            if use_streaming:
                logger.info("Using streaming API for single batch translation")
                # Create generation config with maximum output tokens for streaming
                generation_config = genai.types.GenerateContentConfig(
                    max_output_tokens=8192,  # Request maximum possible output tokens
                    temperature=0.1,  # Lower temperature for more deterministic output
                )

                response = client.models.generate_content_stream(
                    model=model, contents=full_prompt, config=generation_config
                )
            else:
                # Create generation config with maximum output size
                generation_config = genai.types.GenerateContentConfig(
                    max_output_tokens=8192,  # Request maximum possible output tokens
                    temperature=0.1,  # Lower temperature for more deterministic output
                )

                response = client.models.generate_content(
                    model=model, contents=full_prompt, config=generation_config
                )
            translation_time = time.time() - start_time
            logger.info(f"Translation completed in {translation_time:.2f} seconds")

            # If using streaming, response will be different
            if use_streaming:
                logger.debug("Using streaming API for response")

                # For streaming, we need to collect the chunks
                response_chunks = []
                chunk_count = 0

                for chunk in response:
                    chunk_count += 1
                    if hasattr(chunk, "text"):
                        response_chunks.append(chunk.text)
                        logger.debug(
                            f"Received chunk {chunk_count} with {len(chunk.text)} chars"
                        )
                    else:
                        logger.debug(
                            f"Received chunk {chunk_count} without text property"
                        )
                        # Try to extract from candidates
                        if hasattr(chunk, "candidates") and chunk.candidates:
                            for candidate in chunk.candidates:
                                if hasattr(candidate, "content") and hasattr(
                                    candidate.content, "parts"
                                ):
                                    for part in candidate.content.parts:
                                        if hasattr(part, "text"):
                                            response_chunks.append(part.text)
                                            logger.debug(
                                                f"Extracted {len(part.text)} chars from chunk {chunk_count} candidate"
                                            )

                # Combine all streaming chunks
                full_response = "".join(response_chunks)
                logger.debug(
                    f"Combined {chunk_count} streaming chunks into response of {len(full_response)} chars"
                )

                if not full_response:
                    logger.warning("Streaming response is empty")

                return full_response
            else:
                # Debug non-streaming response
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response text length: {len(response.text)} chars")
                logger.debug(f"Response text first 100 chars: {response.text[:100]}")
                logger.debug(
                    f"Response text last 100 chars: {response.text[-100:] if len(response.text) > 100 else response.text}"
                )

                # Print full response object structure for debugging
                logger.debug(f"Full response object: {response}")

                # Check for finish reason that might indicate truncation
                if hasattr(response, "candidates") and response.candidates:
                    for i, candidate in enumerate(response.candidates):
                        if hasattr(candidate, "finish_reason"):
                            logger.debug(
                                f"Candidate {i} finish reason: {candidate.finish_reason}"
                            )
                        if hasattr(candidate, "finish_message"):
                            logger.debug(
                                f"Candidate {i} finish message: {candidate.finish_message}"
                            )
                        if hasattr(candidate, "is_blocked"):
                            logger.debug(
                                f"Candidate {i} is_blocked: {candidate.is_blocked}"
                            )

                # Check if response has parts
                if hasattr(response, "parts"):
                    logger.debug(f"Response has {len(response.parts)} parts")
                    for i, part in enumerate(response.parts):
                        logger.debug(f"Part {i} type: {type(part)}")
                        if hasattr(part, "text"):
                            part_text = part.text
                            logger.debug(
                                f"Part {i} text length: {len(part_text)} chars"
                            )
                            logger.debug(
                                f"Part {i} text first 50 chars: {part_text[:50]}"
                            )
                            logger.debug(
                                f"Part {i} text last 50 chars: {part_text[-50:] if len(part_text) > 50 else part_text}"
                            )

                # Check if we need to access response.text through .candidates property
                if (
                    len(response.text) == 0
                    and hasattr(response, "candidates")
                    and response.candidates
                ):
                    logger.warning(
                        "Response.text is empty, trying to extract from candidates"
                    )
                    candidate_texts = []
                    for candidate in response.candidates:
                        if hasattr(candidate, "content") and hasattr(
                            candidate.content, "parts"
                        ):
                            for part in candidate.content.parts:
                                if hasattr(part, "text"):
                                    candidate_texts.append(part.text)
                                    logger.debug(
                                        f"Found candidate text of length {len(part.text)}"
                                    )

                    if candidate_texts:
                        combined_text = "\n".join(candidate_texts)
                        logger.info(
                            f"Extracted text from candidates, total length: {len(combined_text)} chars"
                        )
                        return combined_text

                return response.text
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            sys.exit(1)
    else:
        # Multi-batch translation
        logger.info(f"Translating {len(batches)} batches to {target_language}...")
        translated_batches = []

        for i, batch in enumerate(batches):
            logger.info(f"Translating batch {i + 1}/{len(batches)}...")
            logger.debug(f"Batch {i + 1} length: {len(batch)} chars")
            batch_start_time = time.time()

            # Add batch context to the prompt
            batch_prompt = prompt
            if i > 0:
                batch_prompt += (
                    "\n\nThis is part "
                    + str(i + 1)
                    + " of a longer text. Continue in the same style and terminology as previous parts."
                )

            try:
                full_batch_prompt = batch_prompt + "\n\n" + batch
                logger.debug(
                    f"Batch {i + 1} full prompt length: {len(full_batch_prompt)} chars"
                )

                # Handle streaming vs non-streaming differently
                if use_streaming:
                    # Create generation config with maximum output tokens for streaming
                    generation_config = genai.types.GenerateContentConfig(
                        max_output_tokens=8192,  # Request maximum possible output tokens
                        temperature=0.1,  # Lower temperature for more deterministic output
                    )

                    response = client.models.generate_content_stream(
                        model=model,
                        contents=full_batch_prompt,
                        config=generation_config,
                    )

                    # For streaming, we need to collect the chunks
                    response_chunks = []
                    chunk_count = 0

                    for chunk in response:
                        chunk_count += 1
                        if hasattr(chunk, "text"):
                            response_chunks.append(chunk.text)
                            logger.debug(
                                f"Batch {i + 1} received chunk {chunk_count} with {len(chunk.text)} chars"
                            )
                        else:
                            logger.debug(
                                f"Batch {i + 1} received chunk {chunk_count} without text property"
                            )
                            # Try to extract from candidates
                            if hasattr(chunk, "candidates") and chunk.candidates:
                                for candidate in chunk.candidates:
                                    if hasattr(candidate, "content") and hasattr(
                                        candidate.content, "parts"
                                    ):
                                        for part in candidate.content.parts:
                                            if hasattr(part, "text"):
                                                response_chunks.append(part.text)
                                                logger.debug(
                                                    f"Batch {i + 1} extracted {len(part.text)} chars from chunk {chunk_count}"
                                                )

                    # Combine all streaming chunks
                    batch_text = "".join(response_chunks)
                    logger.debug(
                        f"Batch {i + 1} combined {chunk_count} streaming chunks into {len(batch_text)} chars"
                    )

                    batch_time = time.time() - batch_start_time
                    logger.info(
                        f"Batch {i + 1} streaming completed in {batch_time:.2f} seconds"
                    )

                    if not batch_text:
                        logger.warning(f"Batch {i + 1} streaming response is empty")

                else:
                    # Non-streaming request
                    # Create generation config with maximum output tokens
                    generation_config = genai.types.GenerateContentConfig(
                        max_output_tokens=8192,  # Request maximum possible output tokens
                        temperature=0.1,  # Lower temperature for more deterministic output
                    )

                    response = client.models.generate_content(
                        model=model,
                        contents=full_batch_prompt,
                        config=generation_config,
                    )
                    batch_time = time.time() - batch_start_time
                    logger.info(f"Batch {i + 1} completed in {batch_time:.2f} seconds")

                    # Debug response for this batch
                    logger.debug(
                        f"Batch {i + 1} response text length: {len(response.text)} chars"
                    )
                    logger.debug(
                        f"Batch {i + 1} response text first 100 chars: {response.text[:100]}"
                    )
                    logger.debug(
                        f"Batch {i + 1} response text last 100 chars: {response.text[-100:] if len(response.text) > 100 else response.text}"
                    )

                    # Check for finish reason that might indicate truncation
                    if hasattr(response, "candidates") and response.candidates:
                        for j, candidate in enumerate(response.candidates):
                            if hasattr(candidate, "finish_reason"):
                                logger.debug(
                                    f"Batch {i + 1} candidate {j} finish reason: {candidate.finish_reason}"
                                )
                            if hasattr(candidate, "finish_message"):
                                logger.debug(
                                    f"Batch {i + 1} candidate {j} finish message: {candidate.finish_message}"
                                )
                            if hasattr(candidate, "is_blocked"):
                                logger.debug(
                                    f"Batch {i + 1} candidate {j} is_blocked: {candidate.is_blocked}"
                                )

                    # Check if response has parts
                    if hasattr(response, "parts"):
                        logger.debug(
                            f"Batch {i + 1} response has {len(response.parts)} parts"
                        )
                        for j, part in enumerate(response.parts):
                            logger.debug(f"Batch {i + 1} part {j} type: {type(part)}")
                            if hasattr(part, "text"):
                                part_text = part.text
                                logger.debug(
                                    f"Batch {i + 1} part {j} text length: {len(part_text)} chars"
                                )

                    # Check if we need to access response.text through .candidates property
                    batch_text = response.text
                    if (
                        len(batch_text) == 0
                        and hasattr(response, "candidates")
                        and response.candidates
                    ):
                        logger.warning(
                            f"Batch {i + 1} response.text is empty, trying to extract from candidates"
                        )
                        candidate_texts = []
                        for candidate in response.candidates:
                            if hasattr(candidate, "content") and hasattr(
                                candidate.content, "parts"
                            ):
                                for part in candidate.content.parts:
                                    if hasattr(part, "text"):
                                        candidate_texts.append(part.text)
                                        logger.debug(
                                            f"Batch {i + 1} found candidate text of length {len(part.text)}"
                                        )

                        if candidate_texts:
                            batch_text = "\n".join(candidate_texts)
                            logger.info(
                                f"Batch {i + 1} extracted text from candidates, total length: {len(batch_text)} chars"
                            )

                # Log batch translation ratio for debugging truncation issues
                batch_ratio = len(batch_text) / len(batch) if len(batch) > 0 else 0
                logger.debug(
                    f"Batch {i + 1} translation ratio: {batch_ratio:.2f} (original: {len(batch)} chars, translated: {len(batch_text)} chars)"
                )

                translated_batches.append(batch_text)
            except Exception as e:
                logger.error(f"Translation of batch {i + 1} failed: {e}")
                logger.error(f"Exception type: {type(e)}")
                sys.exit(1)

        # Debug the combined results
        combined_text = "\n\n".join(translated_batches)
        logger.debug(f"Combined translated text length: {len(combined_text)} chars")
        logger.debug(
            f"Original batches total length: {sum(len(b) for b in batches)} chars"
        )
        logger.debug(f"Number of translated batches: {len(translated_batches)}")

        # Log length of each translated batch
        for i, batch in enumerate(translated_batches):
            logger.debug(f"Translated batch {i + 1} length: {len(batch)} chars")

        return combined_text


def save_outputs(
    original_text: str,
    translated_text: str,
    output_base_path: str,
    formats: List[str] = ["txt", "md", "json"],
    fix_linebreaks: bool = True,
) -> List[str]:
    """
    Save the original and translated text in multiple formats.

    Args:
        original_text: The original text from the file
        translated_text: The translated text
        output_base_path: Base path for output files
        formats: List of formats to save (txt, md, json, csv)
        fix_linebreaks: Whether to fix line breaks in the output

    Returns:
        List of paths to the saved output files
    """
    output_files = []
    output_base = Path(output_base_path)

    # Debug incoming text
    logger.debug(f"save_outputs: Original text length: {len(original_text)} chars")
    logger.debug(f"save_outputs: Translated text length: {len(translated_text)} chars")
    logger.debug(
        f"save_outputs: Translated text first 100 chars: {translated_text[:100]}"
    )
    logger.debug(
        f"save_outputs: Translated text last 100 chars: {translated_text[-100:] if len(translated_text) > 100 else translated_text}"
    )

    # Apply line break fixing if requested
    if fix_linebreaks:
        logger.info("Fixing line breaks in translated text")
        # Fix common line break issues in translated text
        translated_text = re.sub(r"(?<!\n)\n(?!\n)", " ", translated_text)
        # Remove multiple spaces
        translated_text = re.sub(r" +", " ", translated_text)
        # Ensure consistent paragraph breaks
        translated_text = re.sub(r"\n{3,}", "\n\n", translated_text)

        # Debug after fixing line breaks
        logger.debug(
            f"save_outputs: After fixing line breaks - translated text length: {len(translated_text)} chars"
        )

    for fmt in formats:
        if fmt == "txt":
            # Plain text format
            output_path = f"{output_base}_translated.txt"
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(translated_text)
                # Debug file size after writing
                file_size = Path(output_path).stat().st_size
                logger.debug(f"TXT file size: {file_size} bytes")
                output_files.append(output_path)
                logger.info(f"Saved translated text to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save to TXT format: {e}")
                logger.error(f"Exception type: {type(e)}")

        elif fmt == "md":
            # Markdown format with original and translated text
            output_path = f"{output_base}_translated.md"
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("# Original Text\n\n")
                    f.write(original_text)
                    f.write("\n\n# Translated Text\n\n")
                    f.write(translated_text)
                # Debug file size after writing
                file_size = Path(output_path).stat().st_size
                logger.debug(f"MD file size: {file_size} bytes")
                output_files.append(output_path)
                logger.info(f"Saved markdown to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save to MD format: {e}")
                logger.error(f"Exception type: {type(e)}")

        elif fmt == "json":
            # JSON format with original and translated text
            output_path = f"{output_base}_translated.json"
            try:
                data = {
                    "original": original_text,
                    "translated": translated_text,
                    "target_language": args.language,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                # Debug JSON data size
                json_str = json.dumps(data, ensure_ascii=False)
                logger.debug(f"JSON string length: {len(json_str)} chars")

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                # Debug file size after writing
                file_size = Path(output_path).stat().st_size
                logger.debug(f"JSON file size: {file_size} bytes")
                output_files.append(output_path)
                logger.info(f"Saved JSON to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save to JSON format: {e}")
                logger.error(f"Exception type: {type(e)}")
                logger.error(f"JSON error details: {str(e)}")

        elif fmt == "csv":
            # CSV format with original and translated text by paragraph
            output_path = f"{output_base}_translated.csv"
            try:
                # Split text into paragraphs for parallel comparison
                original_paragraphs = [
                    p for p in original_text.split("\n\n") if p.strip()
                ]
                translated_paragraphs = [
                    p for p in translated_text.split("\n\n") if p.strip()
                ]

                # Debug paragraph counts
                logger.debug(f"Original paragraphs: {len(original_paragraphs)}")
                logger.debug(f"Translated paragraphs: {len(translated_paragraphs)}")

                # Ensure both lists have the same length
                max_length = max(len(original_paragraphs), len(translated_paragraphs))
                if len(original_paragraphs) < max_length:
                    original_paragraphs.extend(
                        [""] * (max_length - len(original_paragraphs))
                    )
                if len(translated_paragraphs) < max_length:
                    translated_paragraphs.extend(
                        [""] * (max_length - len(translated_paragraphs))
                    )

                with open(output_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Original", "Translated"])
                    for orig, trans in zip(original_paragraphs, translated_paragraphs):
                        writer.writerow([orig, trans])

                # Debug file size after writing
                file_size = Path(output_path).stat().st_size
                logger.debug(f"CSV file size: {file_size} bytes")
                output_files.append(output_path)
                logger.info(f"Saved CSV to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save to CSV format: {e}")
                logger.error(f"Exception type: {type(e)}")

    return output_files


def main():
    """Parse command line arguments and run the translation process"""
    parser = argparse.ArgumentParser(
        description="Translate various file formats (PDF, LaTeX, Markdown, Text) while preserving formatting"
    )
    parser.add_argument("input_file", help="Path to the input file to translate")
    parser.add_argument(
        "-l",
        "--language",
        default="French",
        help="Target language for translation (default: French)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Base name for output files (default: same as input with _translated suffix)",
    )
    parser.add_argument(
        "-f",
        "--formats",
        default="txt,md,json",
        help="Comma-separated list of output formats: txt,md,json,csv (default: txt,md,json)",
    )
    parser.add_argument(
        "--fix-linebreaks",
        action="store_true",
        default=True,
        help="Fix line breaks in translated output (default: True)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="Custom prompt to use for translation instead of the default",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gemini-2.0-flash",
        help="Gemini model to use for translation (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=MAX_TOKENS,
        help=f"Maximum tokens per batch (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Use streaming API for potentially more complete responses",
    )

    global args
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

    # Detect file type and get appropriate prompt
    file_type, prompt_template = detect_file_type(str(input_path))
    logger.info(f"Detected file type: {file_type}")

    # Extract or read content based on file type
    if file_type == "pdf":
        content = extract_text_from_pdf(str(input_path))
    elif file_type == "latex":
        content = read_latex_file(str(input_path))
    elif file_type == "markdown":
        content = read_markdown_file(str(input_path))
    elif file_type == "text":
        content = read_text_file(str(input_path))
    else:
        logger.error(f"Unsupported file type: {file_type}")
        sys.exit(1)

    # Set output base path
    if args.output:
        output_base = args.output
    else:
        output_base = str(input_path.with_suffix(""))

    # Parse formats
    formats = [fmt.strip().lower() for fmt in args.formats.split(",")]
    valid_formats = ["txt", "md", "json", "csv"]
    for fmt in formats:
        if fmt not in valid_formats:
            logger.warning(f"Invalid format: {fmt}. Will be ignored.")
    formats = [fmt for fmt in formats if fmt in valid_formats]

    # Translate the content
    translated_text = translate_content(
        content=content,
        prompt_template=prompt_template,
        target_language=args.language,
        custom_prompt=args.prompt,
        model=args.model,
        batch_size=args.batch_size,
        use_streaming=args.streaming,
    )

    # Log token count information
    total_tokens = estimate_token_count(content)
    logger.info(f"Estimated total tokens: {total_tokens}")
    if total_tokens > args.batch_size:
        logger.info(
            f"Content was processed in batches (max {args.batch_size} tokens per batch)"
        )

    # Save outputs in requested formats
    output_files = save_outputs(
        original_text=content,
        translated_text=translated_text,
        output_base_path=output_base,
        formats=formats,
        fix_linebreaks=args.fix_linebreaks,
    )

    logger.info(f"Translation complete. Files saved: {', '.join(output_files)}")


if __name__ == "__main__":
    main()
