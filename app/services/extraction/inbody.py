"""InBody body composition scan extractor."""

import re
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.extraction.base import BaseExtractor, ExtractionResult
from app.services.extraction.validation import validate_range

logger = get_logger(__name__)


class InBodyData(BaseModel):
    """Structured InBody extraction results."""

    scan_date: date | None = None
    device_model: str | None = None

    # Core metrics with confidence
    weight_kg: float
    weight_confidence: float = Field(ge=0.0, le=1.0)

    body_fat_percent: float
    body_fat_confidence: float = Field(ge=0.0, le=1.0)

    skeletal_muscle_mass_kg: float
    smm_confidence: float = Field(ge=0.0, le=1.0)

    basal_metabolic_rate_kcal: int
    bmr_confidence: float = Field(ge=0.0, le=1.0)

    # Optional metrics
    body_water_percent: float | None = None
    visceral_fat_level: int | None = None
    lean_body_mass_kg: float | None = None
    bone_mineral_content_kg: float | None = None
    total_body_water_l: float | None = None
    fat_free_mass_kg: float | None = None

    # Segmental analysis (if available)
    segmental_lean: dict[str, float] | None = None
    segmental_fat: dict[str, float] | None = None


class InBodyExtractor(BaseExtractor):
    """
    Extract body composition data from InBody scan PDFs.

    Uses IBM Docling for table and text extraction, with regex pattern
    matching for field identification. Supports InBody 570, 770, and S10 models.
    """

    document_type = "inbody"

    # Supported InBody models
    SUPPORTED_MODELS = ["InBody 570", "InBody 770", "InBody S10", "InBody 230", "InBody 270"]

    # Model detection patterns
    MODEL_PATTERNS = [
        re.compile(r"InBody\s*(570|770|S10|230|270)", re.IGNORECASE),
        re.compile(r"(570|770|S10|230|270)\s*InBody", re.IGNORECASE),
    ]

    # Field patterns for extraction - multiple patterns per field for robustness
    FIELD_PATTERNS: dict[str, list[re.Pattern]] = {
        "weight": [
            re.compile(r"(?:Body\s*Weight|Weight)\s*[:\s]*(\d+\.?\d*)\s*(kg|lbs?)", re.IGNORECASE),
            re.compile(r"(\d+\.?\d*)\s*(kg|lb)\s*(?:Body\s*Weight)", re.IGNORECASE),
            re.compile(r"Weight\s*\(kg\)\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
            re.compile(r"^Weight\s+(\d+\.?\d*)\s+kg", re.MULTILINE | re.IGNORECASE),
        ],
        "body_fat_percent": [
            re.compile(r"(?:Percent\s*Body\s*Fat|PBF|Body\s*Fat\s*%?)\s*[:\s]*(\d+\.?\d*)\s*%?", re.IGNORECASE),
            re.compile(r"(\d+\.?\d*)\s*%?\s*(?:Body\s*Fat|PBF)", re.IGNORECASE),
            re.compile(r"PBF\s*\(%\)\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
            re.compile(r"Body\s*Fat\s*\(%\)\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        ],
        "smm": [
            re.compile(r"(?:Skeletal\s*Muscle\s*Mass|SMM)\s*[:\s]*(\d+\.?\d*)\s*(kg|lbs?)?", re.IGNORECASE),
            re.compile(r"(\d+\.?\d*)\s*(kg|lb)?\s*(?:Skeletal\s*Muscle\s*Mass|SMM)", re.IGNORECASE),
            re.compile(r"SMM\s*\(kg\)\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        ],
        "bmr": [
            re.compile(r"(?:Basal\s*Metabolic\s*Rate|BMR)\s*[:\s]*(\d+)\s*(?:kcal)?", re.IGNORECASE),
            re.compile(r"(\d{3,4})\s*kcal\s*(?:Basal\s*Metabolic\s*Rate|BMR)", re.IGNORECASE),
            re.compile(r"BMR\s*\(kcal\)\s*[:\s]*(\d+)", re.IGNORECASE),
        ],
        "body_water_percent": [
            re.compile(r"(?:Percent\s*Body\s*Water|Body\s*Water\s*%?)\s*[:\s]*(\d+\.?\d*)\s*%?", re.IGNORECASE),
            re.compile(r"TBW\s*%\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        ],
        "visceral_fat": [
            re.compile(r"(?:Visceral\s*Fat\s*(?:Level|Area)?)\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
            re.compile(r"VFL\s*[:\s]*(\d+)", re.IGNORECASE),
        ],
        "lean_body_mass": [
            re.compile(r"(?:Lean\s*Body\s*Mass|LBM)\s*[:\s]*(\d+\.?\d*)\s*(kg|lbs?)?", re.IGNORECASE),
            re.compile(r"LBM\s*\(kg\)\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        ],
        "total_body_water": [
            re.compile(r"(?:Total\s*Body\s*Water|TBW)\s*[:\s]*(\d+\.?\d*)\s*(L|kg)?", re.IGNORECASE),
            re.compile(r"TBW\s*\(L\)\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        ],
        "fat_free_mass": [
            re.compile(r"(?:Fat[\s\-]?Free\s*Mass|FFM)\s*[:\s]*(\d+\.?\d*)\s*(kg|lbs?)?", re.IGNORECASE),
            re.compile(r"FFM\s*\(kg\)\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        ],
        "bone_mineral": [
            re.compile(r"(?:Bone\s*Mineral\s*Content|BMC)\s*[:\s]*(\d+\.?\d*)\s*(kg|lbs?)?", re.IGNORECASE),
            re.compile(r"BMC\s*\(kg\)\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        ],
    }

    # Date patterns
    DATE_PATTERNS = [
        re.compile(r"(?:Test\s*)?Date\s*[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", re.IGNORECASE),
        re.compile(r"(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})", re.IGNORECASE),
        re.compile(r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})", re.IGNORECASE),
    ]

    # Segmental patterns (for body segment analysis)
    SEGMENT_PATTERNS = {
        "left_arm": re.compile(r"L(?:eft)?\s*Arm\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        "right_arm": re.compile(r"R(?:ight)?\s*Arm\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        "trunk": re.compile(r"Trunk\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        "left_leg": re.compile(r"L(?:eft)?\s*Leg\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
        "right_leg": re.compile(r"R(?:ight)?\s*Leg\s*[:\s]*(\d+\.?\d*)", re.IGNORECASE),
    }

    async def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract body composition data from InBody PDF.

        Args:
            file_path: Path to the InBody PDF file

        Returns:
            ExtractionResult with InBodyData or errors
        """
        import time

        start_time = time.time()
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # Use Docling for PDF extraction
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(file_path)

            # Get all text content
            full_text = self._get_full_text(result)

            if not full_text:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=["Could not extract text from PDF"],
                    warnings=[],
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                )

            # Detect InBody model
            device_model = self._detect_model(full_text)
            if device_model:
                logger.info("Detected InBody model", model=device_model)
            else:
                warnings.append("Could not detect InBody model")

            # Extract scan date
            scan_date = self._extract_date(full_text)
            if not scan_date:
                warnings.append("Could not extract scan date")

            # Extract metrics
            extracted: dict[str, Any] = {
                "scan_date": scan_date,
                "device_model": device_model,
            }
            confidences: dict[str, float] = {}

            # Extract core metrics (required)
            weight, weight_conf = self._extract_field(full_text, "weight")
            if weight is not None:
                # Convert to kg if needed
                extracted["weight_kg"] = weight
                extracted["weight_confidence"] = weight_conf
                confidences["weight"] = weight_conf
            else:
                errors.append("Could not extract weight")

            bf, bf_conf = self._extract_field(full_text, "body_fat_percent")
            if bf is not None:
                extracted["body_fat_percent"] = bf
                extracted["body_fat_confidence"] = bf_conf
                confidences["body_fat"] = bf_conf
            else:
                errors.append("Could not extract body fat percentage")

            smm, smm_conf = self._extract_field(full_text, "smm")
            if smm is not None:
                extracted["skeletal_muscle_mass_kg"] = smm
                extracted["smm_confidence"] = smm_conf
                confidences["smm"] = smm_conf
            else:
                errors.append("Could not extract skeletal muscle mass")

            bmr, bmr_conf = self._extract_field(full_text, "bmr")
            if bmr is not None:
                extracted["basal_metabolic_rate_kcal"] = int(bmr)
                extracted["bmr_confidence"] = bmr_conf
                confidences["bmr"] = bmr_conf
            else:
                errors.append("Could not extract basal metabolic rate")

            # Extract optional metrics
            bw_pct, _ = self._extract_field(full_text, "body_water_percent")
            if bw_pct is not None:
                extracted["body_water_percent"] = bw_pct

            vfl, _ = self._extract_field(full_text, "visceral_fat")
            if vfl is not None:
                extracted["visceral_fat_level"] = int(vfl)

            lbm, _ = self._extract_field(full_text, "lean_body_mass")
            if lbm is not None:
                extracted["lean_body_mass_kg"] = lbm

            tbw, _ = self._extract_field(full_text, "total_body_water")
            if tbw is not None:
                extracted["total_body_water_l"] = tbw

            ffm, _ = self._extract_field(full_text, "fat_free_mass")
            if ffm is not None:
                extracted["fat_free_mass_kg"] = ffm

            bmc, _ = self._extract_field(full_text, "bone_mineral")
            if bmc is not None:
                extracted["bone_mineral_content_kg"] = bmc

            # Try to extract segmental data
            segmental = self._extract_segmental(full_text)
            if segmental:
                extracted["segmental_lean"] = segmental

            # Check if we have minimum required fields
            required_fields = ["weight_kg", "body_fat_percent", "skeletal_muscle_mass_kg", "basal_metabolic_rate_kcal"]
            missing_required = [f for f in required_fields if f not in extracted]

            if missing_required:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=errors,
                    warnings=warnings,
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                )

            # Validate extracted data
            is_valid, validation_errors = self.validate(extracted)
            errors.extend(validation_errors)

            # Calculate overall confidence
            overall_confidence = self.calculate_confidence(confidences)

            # Build response data
            try:
                inbody_data = InBodyData(**extracted)
                data = inbody_data.model_dump(mode="json")
            except Exception as e:
                logger.error("Failed to build InBodyData", error=str(e))
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=[f"Data validation failed: {str(e)}"],
                    warnings=warnings,
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                )

            return ExtractionResult(
                success=is_valid,
                data=data,
                confidence=overall_confidence,
                errors=errors if not is_valid else [],
                warnings=warnings,
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

        except ImportError:
            logger.error("Docling not installed")
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=["Docling library not available"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("InBody extraction failed", error=str(e), exc_info=True)
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"Extraction error: {str(e)}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate extracted InBody data.

        Args:
            data: Extracted data dictionary

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []

        # Validate weight (20-500 kg)
        is_valid, error = validate_range("weight_kg", data.get("weight_kg"))
        if not is_valid and error:
            errors.append(error)

        # Validate body fat (3-60%)
        is_valid, error = validate_range("body_fat_percent", data.get("body_fat_percent"))
        if not is_valid and error:
            errors.append(error)

        # Validate SMM (10-100 kg)
        is_valid, error = validate_range("skeletal_muscle_mass_kg", data.get("skeletal_muscle_mass_kg"))
        if not is_valid and error:
            errors.append(error)

        # Validate BMR (800-4000 kcal)
        is_valid, error = validate_range("basal_metabolic_rate_kcal", data.get("basal_metabolic_rate_kcal"))
        if not is_valid and error:
            errors.append(error)

        # Cross-validate: SMM should be less than weight
        weight = data.get("weight_kg")
        smm = data.get("skeletal_muscle_mass_kg")
        if weight and smm and smm >= weight:
            errors.append(f"SMM ({smm} kg) cannot be greater than weight ({weight} kg)")

        # Cross-validate: body fat + lean mass should be close to total weight
        bf_pct = data.get("body_fat_percent")
        if weight and bf_pct:
            fat_mass = weight * (bf_pct / 100)
            lean_mass = weight - fat_mass
            if smm and smm > lean_mass * 1.1:  # 10% tolerance
                errors.append(f"SMM ({smm} kg) seems inconsistent with lean mass ({lean_mass:.1f} kg)")

        return len(errors) == 0, errors

    def _get_full_text(self, docling_result: Any) -> str:
        """Extract all text from Docling result."""
        texts: list[str] = []

        # Get text from document
        if hasattr(docling_result, "document"):
            doc = docling_result.document

            # Get text from text blocks
            if hasattr(doc, "texts"):
                for text_block in doc.texts:
                    if hasattr(text_block, "text"):
                        texts.append(text_block.text)

            # Get text from tables
            if hasattr(doc, "tables"):
                for table in doc.tables:
                    if hasattr(table, "to_text"):
                        texts.append(table.to_text())
                    elif hasattr(table, "cells"):
                        for row in table.cells:
                            for cell in row:
                                if hasattr(cell, "text"):
                                    texts.append(cell.text)

        # Fallback: try export_to_markdown
        if not texts and hasattr(docling_result, "document"):
            try:
                md = docling_result.document.export_to_markdown()
                texts.append(md)
            except Exception:
                pass

        return "\n".join(texts)

    def _detect_model(self, text: str) -> str | None:
        """Detect InBody device model from text."""
        for pattern in self.MODEL_PATTERNS:
            match = pattern.search(text)
            if match:
                model_num = match.group(1)
                return f"InBody {model_num}"
        return None

    def _extract_date(self, text: str) -> date | None:
        """Extract scan date from text."""
        for pattern in self.DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                date_str = match.group(1)
                parsed = self._parse_date(date_str)
                if parsed and parsed <= date.today():
                    return parsed
        return None

    def _extract_field(
        self,
        text: str,
        field_name: str,
    ) -> tuple[float | None, float]:
        """
        Extract a field value from text using patterns.

        Returns:
            Tuple of (value, confidence)
        """
        patterns = self.FIELD_PATTERNS.get(field_name, [])
        matches: list[float] = []

        for pattern in patterns:
            for match in pattern.finditer(text):
                try:
                    value_str = match.group(1)
                    value = float(value_str)

                    # Check for unit conversion if present
                    if len(match.groups()) > 1 and match.group(2):
                        unit = match.group(2).lower()
                        if field_name in ("weight", "smm", "lean_body_mass", "fat_free_mass", "bone_mineral"):
                            if unit in ("lb", "lbs"):
                                value = value * 0.453592

                    matches.append(value)
                except (ValueError, IndexError):
                    continue

        if not matches:
            return None, 0.0

        # If multiple matches, check for consistency
        if len(matches) == 1:
            return matches[0], 0.9  # Single match - high but not perfect confidence
        elif len(set(matches)) == 1:
            return matches[0], 0.95  # Multiple identical matches - very high confidence
        else:
            # Multiple different values - use the most common, lower confidence
            from collections import Counter

            counter = Counter(matches)
            most_common = counter.most_common(1)[0][0]
            return most_common, 0.7

    def _extract_segmental(self, text: str) -> dict[str, float] | None:
        """Extract segmental analysis data."""
        segments: dict[str, float] = {}

        for segment_name, pattern in self.SEGMENT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                try:
                    segments[segment_name] = float(match.group(1))
                except (ValueError, IndexError):
                    continue

        return segments if segments else None
