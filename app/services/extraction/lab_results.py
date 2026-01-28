"""Lab results extractor for Quest, LabCorp, and generic lab CSV/PDF files."""

import csv
import io
import re
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.extraction.base import BaseExtractor, ExtractionResult
from app.services.extraction.validation import (
    BIOMARKER_REFERENCE_RANGES,
    flag_biomarker_value,
    validate_date_not_future,
)

logger = get_logger(__name__)


class BiomarkerValue(BaseModel):
    """A single biomarker measurement from lab results."""

    name: str
    code: str | None = None  # LOINC code if available
    value: float
    unit: str
    reference_range: str | None = None
    flag: str | None = None  # "high", "low", "normal", "critical"
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)


class LabResultsData(BaseModel):
    """Structured data extracted from lab results."""

    lab_provider: str | None = None
    collection_date: date | None = None
    report_date: date | None = None
    biomarkers: list[BiomarkerValue]


class LabResultsExtractor(BaseExtractor):
    """
    Extract biomarkers from lab result documents.

    Uses configurable document extraction providers (Docling, Reducto, etc.)
    for PDF extraction, with direct parsing for CSV files.

    Supports:
    - Quest Diagnostics PDF and CSV
    - LabCorp PDF and CSV
    - Generic lab CSV with common column formats
    """

    document_type = "lab_results"

    # Lab provider detection patterns
    PROVIDER_PATTERNS = {
        "Quest Diagnostics": [
            re.compile(r"Quest\s*Diagnostics", re.IGNORECASE),
            re.compile(r"questdiagnostics\.com", re.IGNORECASE),
        ],
        "LabCorp": [
            re.compile(r"LabCorp|Laboratory\s*Corporation", re.IGNORECASE),
            re.compile(r"labcorp\.com", re.IGNORECASE),
        ],
    }

    # Common biomarker name patterns and their standardized names
    BIOMARKER_ALIASES: dict[str, list[str]] = {
        "total_cholesterol": ["total cholesterol", "cholesterol, total", "tc"],
        "ldl_cholesterol": ["ldl cholesterol", "ldl-c", "ldl", "ldl chol", "low density lipoprotein"],
        "hdl_cholesterol": ["hdl cholesterol", "hdl-c", "hdl", "hdl chol", "high density lipoprotein"],
        "triglycerides": ["triglycerides", "trig", "trigs", "triglyceride"],
        "glucose": ["glucose", "fasting glucose", "blood glucose", "glucose, fasting", "gluc"],
        "hemoglobin_a1c": ["hemoglobin a1c", "hba1c", "a1c", "glycated hemoglobin", "hgb a1c"],
        "creatinine": ["creatinine", "creat", "serum creatinine"],
        "bun": ["bun", "blood urea nitrogen", "urea nitrogen"],
        "egfr": ["egfr", "gfr", "estimated gfr", "glomerular filtration rate"],
        "ast": ["ast", "sgot", "aspartate aminotransferase"],
        "alt": ["alt", "sgpt", "alanine aminotransferase"],
        "alkaline_phosphatase": ["alkaline phosphatase", "alk phos", "alp"],
        "bilirubin": ["bilirubin", "total bilirubin", "bili"],
        "albumin": ["albumin", "alb"],
        "total_protein": ["total protein", "protein, total"],
        "calcium": ["calcium", "ca", "serum calcium"],
        "sodium": ["sodium", "na"],
        "potassium": ["potassium", "k"],
        "chloride": ["chloride", "cl"],
        "co2": ["co2", "carbon dioxide", "bicarbonate"],
        "tsh": ["tsh", "thyroid stimulating hormone", "thyrotropin"],
        "t3": ["t3", "triiodothyronine", "free t3", "t3 free"],
        "t4": ["t4", "thyroxine", "free t4", "t4 free"],
        "testosterone": ["testosterone", "total testosterone", "test total"],
        "estradiol": ["estradiol", "e2", "estrogen"],
        "cortisol": ["cortisol", "am cortisol", "serum cortisol"],
        "vitamin_d": ["vitamin d", "25-oh vitamin d", "25-hydroxyvitamin d", "d 25-hydroxy"],
        "vitamin_b12": ["vitamin b12", "b12", "cobalamin"],
        "folate": ["folate", "folic acid"],
        "iron": ["iron", "serum iron", "fe"],
        "ferritin": ["ferritin"],
        "tibc": ["tibc", "total iron binding capacity"],
        "crp": ["crp", "c-reactive protein", "hs-crp", "high sensitivity crp"],
        "esr": ["esr", "sed rate", "erythrocyte sedimentation rate"],
        "wbc": ["wbc", "white blood cell", "white blood count", "leukocytes"],
        "rbc": ["rbc", "red blood cell", "red blood count", "erythrocytes"],
        "hemoglobin": ["hemoglobin", "hgb", "hb"],
        "hematocrit": ["hematocrit", "hct"],
        "platelets": ["platelets", "platelet count", "plt"],
        "psa": ["psa", "prostate specific antigen"],
    }

    # Patterns for extracting values from PDF text
    VALUE_PATTERN = re.compile(
        r"(?P<name>[A-Za-z][A-Za-z0-9\s,\-\(\)]+?)\s*"
        r"(?P<value>\d+\.?\d*)\s*"
        r"(?P<unit>[A-Za-z/%]+(?:/[A-Za-z]+)?)\s*"
        r"(?:(?P<range>[\d\.\-\s<>]+(?:\s*-\s*[\d\.]+)?))?"
    )

    # Date patterns
    DATE_PATTERNS = [
        re.compile(r"(?:Collection|Collected|Draw)\s*(?:Date)?[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", re.IGNORECASE),
        re.compile(r"(?:Report|Reported)\s*(?:Date)?[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", re.IGNORECASE),
        re.compile(r"Date[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", re.IGNORECASE),
    ]

    # CSV column name mappings
    CSV_COLUMN_ALIASES = {
        "name": ["test name", "test", "analyte", "component", "biomarker", "name"],
        "value": ["result", "value", "result value", "your value", "your result"],
        "unit": ["unit", "units", "unit of measure", "uom"],
        "range": ["reference range", "ref range", "normal range", "reference interval", "range"],
        "flag": ["flag", "abnormal", "status", "abnormal flag"],
    }

    async def extract(
        self,
        file_path: str,
        provider: str | None = None,
    ) -> ExtractionResult:
        """
        Extract biomarkers from lab results file.

        Args:
            file_path: Path to the lab results file (PDF or CSV)
            provider: Optional provider name ("docling", "reducto").
                     If not specified, uses the configured default.
                     Only applicable for PDF files.

        Returns:
            ExtractionResult with LabResultsData or errors
        """
        import time

        start_time = time.time()
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            return await self._extract_from_pdf(file_path, start_time, provider)
        elif ext == ".csv":
            return await self._extract_from_csv(file_path, start_time)
        else:
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"Unsupported file type: {ext}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
            )

    async def _extract_from_pdf(
        self,
        file_path: str,
        start_time: float,
        provider: str | None = None,
    ) -> ExtractionResult:
        """Extract biomarkers from PDF using configurable provider."""
        from app.services.extraction.providers import get_provider

        errors: list[str] = []
        warnings: list[str] = []
        used_provider = provider

        try:
            # Get document extraction provider
            doc_provider = get_provider(provider)
            used_provider = doc_provider.name
            logger.info(
                "Using extraction provider",
                provider=used_provider,
                file_path=file_path,
            )

            # Extract document content using provider
            content = await doc_provider.extract(Path(file_path))

            # Use markdown content if available, fallback to plain text
            full_text = content.markdown or content.text

            if not full_text:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=["Could not extract text from PDF"],
                    warnings=[],
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                    metadata={"provider": used_provider},
                )

            # Detect lab provider
            lab_provider = self._detect_provider(full_text)

            # Extract dates
            collection_date = self._extract_date(full_text, "collection")
            report_date = self._extract_date(full_text, "report")

            # Extract biomarkers from tables
            biomarkers = self._extract_biomarkers_from_text(full_text)

            if not biomarkers:
                warnings.append("No biomarkers could be extracted from PDF")

            # Validate dates
            if collection_date:
                is_valid, error = validate_date_not_future("collection_date", collection_date)
                if not is_valid and error:
                    warnings.append(error)

            # Calculate confidence
            confidence = 0.8 if biomarkers else 0.0
            if lab_provider:
                confidence += 0.1
            if collection_date:
                confidence += 0.05

            data = LabResultsData(
                lab_provider=lab_provider,
                collection_date=collection_date,
                report_date=report_date,
                biomarkers=biomarkers,
            ).model_dump(mode="json")

            return ExtractionResult(
                success=len(biomarkers) > 0,
                data=data,
                confidence=min(confidence, 1.0),
                errors=errors,
                warnings=warnings,
                extraction_time_ms=int((time.time() - start_time) * 1000),
                metadata={"provider": used_provider},
            )

        except Exception as e:
            logger.error(
                "Lab results PDF extraction failed",
                error=str(e),
                provider=used_provider,
                exc_info=True,
            )

            # Try fallback to docling if we were using a different provider
            if provider and provider.lower() != "docling":
                logger.info("Attempting fallback to docling provider")
                try:
                    return await self._extract_from_pdf(file_path, start_time, provider="docling")
                except Exception as fallback_error:
                    logger.error("Fallback extraction also failed", error=str(fallback_error))

            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"Extraction error: {str(e)}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
                metadata={"provider": used_provider or "unknown"},
            )

    async def _extract_from_csv(self, file_path: str, start_time: float) -> ExtractionResult:
        """Extract biomarkers from CSV file (no provider needed)."""
        errors: list[str] = []
        warnings: list[str] = []

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                content = f.read()

            # Detect delimiter
            delimiter = self._detect_csv_delimiter(content)

            # Parse CSV
            reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
            rows = list(reader)

            if not rows:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=["CSV file is empty"],
                    warnings=[],
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                    metadata={"provider": "csv_parser"},
                )

            # Map column names to standard names
            column_map = self._map_csv_columns(reader.fieldnames or [])

            if "name" not in column_map or "value" not in column_map:
                return ExtractionResult(
                    success=False,
                    data=None,
                    confidence=0.0,
                    errors=["Could not identify name and value columns in CSV"],
                    warnings=[],
                    extraction_time_ms=int((time.time() - start_time) * 1000),
                    metadata={"provider": "csv_parser"},
                )

            # Extract biomarkers
            biomarkers: list[BiomarkerValue] = []

            for row in rows:
                try:
                    name_col = column_map.get("name")
                    value_col = column_map.get("value")
                    unit_col = column_map.get("unit")
                    range_col = column_map.get("range")
                    flag_col = column_map.get("flag")

                    if not name_col or not value_col:
                        continue

                    raw_name = row.get(name_col, "").strip()
                    raw_value = row.get(value_col, "").strip()

                    if not raw_name or not raw_value:
                        continue

                    # Parse value (handle < or > prefixes)
                    value = self._parse_numeric_value(raw_value)
                    if value is None:
                        continue

                    # Standardize name
                    std_name = self._standardize_biomarker_name(raw_name)

                    # Get unit
                    unit = row.get(unit_col, "").strip() if unit_col else ""

                    # Get reference range
                    ref_range = row.get(range_col, "").strip() if range_col else None

                    # Get or determine flag
                    flag = row.get(flag_col, "").strip().lower() if flag_col else None
                    if not flag and std_name:
                        flag = flag_biomarker_value(std_name, value)

                    biomarkers.append(
                        BiomarkerValue(
                            name=raw_name,
                            code=None,
                            value=value,
                            unit=unit or "unknown",
                            reference_range=ref_range,
                            flag=flag,
                            confidence=0.95,  # CSV parsing is generally reliable
                        )
                    )

                except Exception as e:
                    warnings.append(f"Failed to parse row: {str(e)}")
                    continue

            if not biomarkers:
                errors.append("No valid biomarkers found in CSV")

            data = LabResultsData(
                lab_provider=None,  # Can't detect from CSV usually
                collection_date=None,
                report_date=None,
                biomarkers=biomarkers,
            ).model_dump(mode="json")

            return ExtractionResult(
                success=len(biomarkers) > 0,
                data=data,
                confidence=0.9 if biomarkers else 0.0,
                errors=errors,
                warnings=warnings,
                extraction_time_ms=int((time.time() - start_time) * 1000),
                metadata={"provider": "csv_parser"},
            )

        except Exception as e:
            logger.error("Lab results CSV extraction failed", error=str(e), exc_info=True)
            return ExtractionResult(
                success=False,
                data=None,
                confidence=0.0,
                errors=[f"CSV parsing error: {str(e)}"],
                warnings=[],
                extraction_time_ms=int((time.time() - start_time) * 1000),
                metadata={"provider": "csv_parser"},
            )

    def validate(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate extracted lab results data."""
        errors: list[str] = []

        biomarkers = data.get("biomarkers", [])

        if not biomarkers:
            errors.append("No biomarkers extracted")
            return False, errors

        # Validate individual biomarker values
        for biomarker in biomarkers:
            name = biomarker.get("name", "unknown")
            value = biomarker.get("value")

            # Check for unreasonable values
            if value is not None and (value < 0 or value > 100000):
                errors.append(f"Unreasonable value for {name}: {value}")

        return len(errors) == 0, errors

    def _detect_provider(self, text: str) -> str | None:
        """Detect lab provider from text."""
        for provider, patterns in self.PROVIDER_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(text):
                    return provider
        return None

    def _extract_date(self, text: str, date_type: str) -> date | None:
        """Extract collection or report date from text."""
        for pattern in self.DATE_PATTERNS:
            if date_type.lower() in pattern.pattern.lower():
                match = pattern.search(text)
                if match:
                    return self._parse_date(match.group(1))
        return None

    def _extract_biomarkers_from_text(self, text: str) -> list[BiomarkerValue]:
        """Extract biomarker values from text using pattern matching."""
        biomarkers: list[BiomarkerValue] = []
        seen_names: set[str] = set()

        # Split into lines and look for patterns
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue

            # Try to match biomarker patterns
            for std_name, aliases in self.BIOMARKER_ALIASES.items():
                for alias in aliases:
                    # Look for alias followed by a number
                    pattern = re.compile(
                        rf"(?:^|\s){re.escape(alias)}\s*[:\s]*(\d+\.?\d*)\s*([A-Za-z/%]+(?:/[A-Za-z]+)?)?",
                        re.IGNORECASE,
                    )
                    match = pattern.search(line)

                    if match:
                        # Avoid duplicates
                        if std_name in seen_names:
                            continue

                        try:
                            value = float(match.group(1))
                            unit = match.group(2) or ""

                            # Get reference range info
                            ref_range = BIOMARKER_REFERENCE_RANGES.get(std_name)
                            if ref_range and not unit:
                                unit = ref_range.get("unit", "")

                            flag = flag_biomarker_value(std_name, value)

                            biomarkers.append(
                                BiomarkerValue(
                                    name=alias.title(),
                                    code=None,
                                    value=value,
                                    unit=unit,
                                    reference_range=None,
                                    flag=flag,
                                    confidence=0.85,
                                )
                            )
                            seen_names.add(std_name)
                        except ValueError:
                            continue

        return biomarkers

    def _detect_csv_delimiter(self, content: str) -> str:
        """Detect the delimiter used in a CSV file."""
        # Count occurrences of common delimiters in first line
        first_line = content.split("\n")[0] if "\n" in content else content
        delimiters = [",", "\t", ";", "|"]

        counts = {d: first_line.count(d) for d in delimiters}
        return max(counts, key=lambda d: counts[d])

    def _map_csv_columns(self, fieldnames: list[str]) -> dict[str, str]:
        """Map CSV column names to standard field names."""
        column_map: dict[str, str] = {}
        lower_fields = {f.lower().strip(): f for f in fieldnames}

        for std_name, aliases in self.CSV_COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in lower_fields:
                    column_map[std_name] = lower_fields[alias]
                    break

        return column_map

    def _standardize_biomarker_name(self, raw_name: str) -> str | None:
        """Convert a raw biomarker name to its standardized form."""
        raw_lower = raw_name.lower().strip()

        for std_name, aliases in self.BIOMARKER_ALIASES.items():
            for alias in aliases:
                if alias == raw_lower or alias in raw_lower:
                    return std_name

        return None

    def _parse_numeric_value(self, value_str: str) -> float | None:
        """Parse a numeric value, handling < and > prefixes."""
        value_str = value_str.strip()

        # Remove < or > prefix
        if value_str.startswith("<") or value_str.startswith(">"):
            value_str = value_str[1:].strip()

        # Remove commas
        value_str = value_str.replace(",", "")

        try:
            return float(value_str)
        except ValueError:
            return None
