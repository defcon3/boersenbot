#!/usr/bin/env python3
"""
Screenshot OCR Parser — Extract Jupiter Prediction Market data from screenshots
Uses EasyOCR to parse event names and quotes (¢ format)
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    logging.warning("easyocr not installed. Install with: pip install easyocr")

from PIL import Image


# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "ocr_parser.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Config
SCREENSHOTS_DIR = Path("C:\\Users\\defco\\Desktop\\screen")
OUTPUT_DIR = Path("jupag_extracted")
OUTPUT_DIR.mkdir(exist_ok=True)


class ScreenshotOCRParser:
    """Extracts Jupiter prediction market data from screenshots"""

    def __init__(self):
        if HAS_EASYOCR:
            logger.info("Initializing OCR reader...")
            self.reader = easyocr.Reader(['en'])
        else:
            self.reader = None
            logger.warning("EasyOCR not available - using fallback parsing")

    def extract_text_from_image(self, image_path: Path) -> str:
        """Extract text from image using OCR"""
        if not self.reader:
            logger.warning(f"OCR reader not available for {image_path}")
            return ""

        try:
            logger.info(f"Processing {image_path.name}...")
            result = self.reader.readtext(str(image_path))

            # Combine all detected text
            full_text = "\n".join([text[1] for text in result])
            return full_text
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return ""

    def parse_wm_matches(self, text: str) -> list:
        """
        Parse WM matches from OCR text
        Looks for patterns like:
        - "Germany vs France" + "66¢" "21¢" "13¢"
        - Event names with country flags or "vs"
        """
        matches = []

        # Split by common delimiters
        lines = text.split("\n")

        # Common tournament indicators
        wm_keywords = ["FIFA", "World Cup", "WM", "Weltmeisterschaft", "Group", "Round"]

        current_match = None
        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line or len(line) < 3:
                continue

            # Check if this is a potential match line (has "vs" or country names)
            if any(kw in line for kw in ["vs", "vs.", "versus", "-"]) and not "¢" in line:
                # This might be a match title
                current_match = {
                    "title": line,
                    "prices": {},
                    "raw_line": line,
                }

            # Look for price lines (contain ¢ symbol)
            if "¢" in line:
                # Extract prices in format "Option 66¢" or just "66¢"
                price_pattern = r"([A-Za-z\s]+)?(\d+)¢"
                found_prices = re.findall(price_pattern, line)

                if found_prices and current_match:
                    for option, price in found_prices:
                        option_name = option.strip() if option else f"option_{len(current_match['prices'])}"
                        price_decimal = int(price) / 100.0
                        current_match["prices"][option_name] = price_decimal

                    # Try to finalize the match if we have 2+ prices
                    if len(current_match["prices"]) >= 2:
                        matches.append(current_match)
                        current_match = None

        return matches

    def parse_screenshots(self):
        """Parse all screenshots in the directory"""
        if not SCREENSHOTS_DIR.exists():
            logger.error(f"Screenshot directory not found: {SCREENSHOTS_DIR}")
            return []

        png_files = sorted(SCREENSHOTS_DIR.glob("*.PNG")) + sorted(SCREENSHOTS_DIR.glob("*.png"))
        logger.info(f"Found {len(png_files)} screenshot(s)")

        all_matches = []

        for png_file in png_files:
            logger.info(f"\n=== Processing {png_file.name} ===")

            # Extract text from image
            text = self.extract_text_from_image(png_file)

            if text:
                # Parse WM matches
                matches = self.parse_wm_matches(text)
                logger.info(f"Extracted {len(matches)} matches from {png_file.name}")

                for match in matches:
                    logger.info(f"  - {match['title']}: {match['prices']}")
                    all_matches.append(match)
            else:
                logger.warning(f"No text extracted from {png_file.name}")

        return all_matches

    def save_matches(self, matches: list, filename: str = "wm_matches.json"):
        """Save extracted matches to JSON"""
        output_file = OUTPUT_DIR / filename

        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "Jupiter Prediction Markets (Screenshot OCR)",
            "total_matches": len(matches),
            "matches": matches,
        }

        try:
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"\nSaved {len(matches)} matches to {output_file}")
        except Exception as e:
            logger.error(f"Error saving matches: {e}")


def main():
    logger.info("Screenshot OCR Parser starting...")

    if not HAS_EASYOCR:
        logger.error("EasyOCR not installed!")
        logger.info("Install with: pip install easyocr")
        return

    parser = ScreenshotOCRParser()
    matches = parser.parse_screenshots()

    if matches:
        logger.info(f"\n=== Summary ===")
        logger.info(f"Total matches extracted: {len(matches)}")

        # Show sample
        logger.info(f"\nSample matches:")
        for m in matches[:5]:
            logger.info(f"  - {m['title']}: {m['prices']}")

        parser.save_matches(matches)
    else:
        logger.warning("No matches extracted from screenshots")


if __name__ == "__main__":
    main()
