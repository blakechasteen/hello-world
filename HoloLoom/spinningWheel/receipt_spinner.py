"""
Receipt Spinner - OCR for Receipt Parsing
==========================================

Specialized spinner for extracting and parsing receipt data.

Features:
- Receipt structure detection (merchant, date, items, total)
- Line item extraction (name, quantity, price)
- Tax and total calculation verification
- Merchant identification
- Payment method detection
- Category classification (grocery, restaurant, retail, etc.)
- Expense tracking metadata

Supports:
- Paper receipts (photos, scans)
- Digital receipts (PDF, image)
- Multiple currencies
- Various receipt layouts

Usage:
    from HoloLoom.spinningWheel import ReceiptSpinner

    spinner = ReceiptSpinner()
    result = await spinner.spin("receipt.jpg")

    # Access structured data
    receipt_data = result.shards[0].metadata['receipt_data']
    print(f"Merchant: {receipt_data['merchant']}")
    print(f"Total: ${receipt_data['total']}")
    print(f"Items: {len(receipt_data['items'])}")

Author: Claude Code
Date: January 2025
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import re
from decimal import Decimal

from HoloLoom.protocols.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    ImportanceScore,
    ImportanceSignals
)
from HoloLoom.spinningWheel.ocr_protocol import OCROutputFormat
from HoloLoom.spinningWheel.ocr_backends import get_all_available_backends


@dataclass
class LineItem:
    """Single line item from receipt"""
    name: str
    quantity: float = 1.0
    unit_price: Optional[Decimal] = None
    total_price: Optional[Decimal] = None
    category: Optional[str] = None


@dataclass
class ReceiptData:
    """Parsed receipt data"""
    merchant: Optional[str] = None
    merchant_address: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None

    items: List[LineItem] = field(default_factory=list)

    subtotal: Optional[Decimal] = None
    tax: Optional[Decimal] = None
    tip: Optional[Decimal] = None
    total: Optional[Decimal] = None

    payment_method: Optional[str] = None
    last_four: Optional[str] = None  # Last 4 digits of card

    category: Optional[str] = None  # grocery, restaurant, retail, etc.
    currency: str = "USD"

    @property
    def item_count(self) -> int:
        """Total number of items."""
        return len(self.items)

    @property
    def total_quantity(self) -> float:
        """Total quantity of all items."""
        return sum(item.quantity for item in self.items)


class ReceiptSpinner(BaseSpinner):
    """
    Specialized spinner for receipt parsing.

    Optimized for:
    - Receipt text extraction
    - Structured data parsing
    - Line item extraction
    - Financial calculation verification
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        verify_calculations: bool = True,
        categorize: bool = True,
        enable_enrichment: bool = False
    ):
        """
        Initialize ReceiptSpinner.

        Args:
            importance_threshold: Min importance score (0.0-1.0)
            verify_calculations: Verify subtotal/tax/total calculations
            categorize: Categorize receipts (grocery, restaurant, etc.)
            enable_enrichment: Extract additional entities via Ollama
        """
        super().__init__(name="receipt", importance_threshold=importance_threshold)

        self.verify_calculations = verify_calculations
        self.categorize = categorize
        self.enable_enrichment = enable_enrichment

        # Get OCR backend chain
        self.ocr_chain = get_all_available_backends()

        print(f"[ReceiptSpinner] Using OCR backend: {self.ocr_chain.get_name()} ({self.ocr_chain.get_quality().value})")

    def get_capabilities(self) -> SpinnerCapabilities:
        """Return spinner capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,
            incremental=False,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=True,  # Merchant, items, dates
            motif_extraction=True,
            embeddings=False,
            multimodal=False,
            supported_formats=['.jpg', '.jpeg', '.png', '.pdf', '.tiff']
        )

    def is_available(self) -> bool:
        """Check if OCR backend is available."""
        return self.ocr_chain.is_available()

    async def _spin_impl(
        self,
        source: Any,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Extract and parse receipt(s).

        Args:
            source: Receipt image path(s) - str, Path, or list
            **kwargs: Additional options

        Returns:
            List of MemoryShards
        """
        # Normalize to list of paths
        if isinstance(source, (str, Path)):
            receipt_paths = [Path(source)]
        elif isinstance(source, list):
            receipt_paths = [Path(p) for p in source]
        else:
            raise ValueError(f"source must be path or list of paths, got {type(source)}")

        # Validate paths
        for path in receipt_paths:
            if not path.exists():
                raise FileNotFoundError(f"Receipt not found: {path}")

        # Process receipts
        all_shards = []

        for receipt_path in receipt_paths:
            # OCR extraction (TEXT format for easier parsing)
            ocr_result = await self.ocr_chain.extract_text(
                receipt_path,
                output_format=OCROutputFormat.TEXT
            )

            # Parse receipt structure
            receipt_data = self._parse_receipt(ocr_result.text)

            # Categorize if enabled
            if self.categorize and not receipt_data.category:
                receipt_data.category = self._categorize_receipt(receipt_data)

            # Verify calculations if enabled
            calculation_verified = False
            if self.verify_calculations:
                calculation_verified = self._verify_calculations(receipt_data)

            # Create shard
            shard = await self._create_shard_from_receipt(
                receipt_path,
                ocr_result.text,
                receipt_data,
                ocr_result,
                calculation_verified
            )

            all_shards.append(shard)

        return all_shards

    async def _create_shard_from_receipt(
        self,
        receipt_path: Path,
        raw_text: str,
        receipt_data: ReceiptData,
        ocr_result: Any,
        calculation_verified: bool
    ) -> MemoryShard:
        """
        Create MemoryShard from parsed receipt.

        Args:
            receipt_path: Path to receipt image
            raw_text: Raw OCR text
            receipt_data: Parsed receipt data
            ocr_result: OCRResult from backend
            calculation_verified: Whether calculations check out

        Returns:
            MemoryShard
        """
        # Format receipt text for display
        formatted_text = self._format_receipt_text(receipt_data, raw_text)

        # Extract entities
        entities = self._extract_receipt_entities(receipt_data)

        # Extract motifs
        motifs = self._extract_receipt_motifs(receipt_data)

        # Enrich if enabled
        if self.enable_enrichment and len(raw_text) > 50:
            enriched_entities, enriched_motifs = await self._enrich_content(raw_text)
            entities.extend(enriched_entities)
            motifs.extend(enriched_motifs)

        # Deduplicate
        entities = list(set(entities))[:20]
        motifs = list(set(motifs))[:10]

        # Score importance
        importance = self._score_receipt_importance(receipt_data, ocr_result.confidence)

        # Create shard
        shard = self._create_shard(
            id_suffix=f"{receipt_path.stem}_{int(time.time())}",
            text=formatted_text,
            episode=f"receipt_{receipt_path.stem}",
            entities=entities,
            motifs=motifs,
            metadata={
                'file_path': str(receipt_path),
                'file_name': receipt_path.name,
                'file_size': receipt_path.stat().st_size,
                'ocr_backend': ocr_result.backend,
                'ocr_confidence': ocr_result.confidence,
                'ocr_quality': ocr_result.quality.value,
                'document_type': 'receipt',

                # Receipt data (structured)
                'receipt_data': {
                    'merchant': receipt_data.merchant,
                    'merchant_address': receipt_data.merchant_address,
                    'date': receipt_data.date,
                    'time': receipt_data.time,
                    'subtotal': float(receipt_data.subtotal) if receipt_data.subtotal else None,
                    'tax': float(receipt_data.tax) if receipt_data.tax else None,
                    'tip': float(receipt_data.tip) if receipt_data.tip else None,
                    'total': float(receipt_data.total) if receipt_data.total else None,
                    'payment_method': receipt_data.payment_method,
                    'last_four': receipt_data.last_four,
                    'category': receipt_data.category,
                    'currency': receipt_data.currency,
                    'item_count': receipt_data.item_count,
                    'total_quantity': receipt_data.total_quantity,
                    'items': [
                        {
                            'name': item.name,
                            'quantity': item.quantity,
                            'unit_price': float(item.unit_price) if item.unit_price else None,
                            'total_price': float(item.total_price) if item.total_price else None,
                            'category': item.category
                        }
                        for item in receipt_data.items
                    ]
                },

                'calculation_verified': calculation_verified,
                'importance_score': importance.score,
                'importance_reason': importance.reason
            }
        )

        return shard

    def _parse_receipt(self, text: str) -> ReceiptData:
        """
        Parse receipt text into structured data.

        Args:
            text: Raw OCR text

        Returns:
            ReceiptData object
        """
        receipt = ReceiptData()
        lines = text.split('\n')

        # Extract merchant (usually first few lines, capitalized)
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if len(line) > 3 and line.isupper():
                receipt.merchant = line
                break

        # Extract date (various formats)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                receipt.date = match.group(0)
                break

        # Extract time
        time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\b'
        time_match = re.search(time_pattern, text, re.IGNORECASE)
        if time_match:
            receipt.time = time_match.group(0)

        # Extract totals (look for keywords)
        amount_pattern = r'(\$?\d+[,\.]?\d*\.?\d{2})'

        # Subtotal
        subtotal_match = re.search(rf'(?:subtotal|sub\s*total)[:\s]*{amount_pattern}', text, re.IGNORECASE)
        if subtotal_match:
            receipt.subtotal = self._parse_decimal(subtotal_match.group(1))

        # Tax
        tax_match = re.search(rf'(?:tax|gst|vat)[:\s]*{amount_pattern}', text, re.IGNORECASE)
        if tax_match:
            receipt.tax = self._parse_decimal(tax_match.group(1))

        # Tip
        tip_match = re.search(rf'(?:tip|gratuity)[:\s]*{amount_pattern}', text, re.IGNORECASE)
        if tip_match:
            receipt.tip = self._parse_decimal(tip_match.group(1))

        # Total (multiple patterns)
        total_patterns = [
            rf'(?:total|amount\s*due|balance)[:\s]*{amount_pattern}',
            rf'(?:grand\s*total)[:\s]*{amount_pattern}'
        ]

        for pattern in total_patterns:
            total_match = re.search(pattern, text, re.IGNORECASE)
            if total_match:
                receipt.total = self._parse_decimal(total_match.group(1))
                break

        # Extract line items (heuristic: line with price at end)
        item_pattern = r'^(.+?)\s+(\d*\.?\d+)\s*$'  # Item name followed by price

        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # Look for line ending in price
            match = re.match(r'^(.+?)\s+(\$?\d+[,\.]?\d*\.?\d{2})$', line)
            if match:
                item_name = match.group(1).strip()
                item_price = self._parse_decimal(match.group(2))

                # Skip if looks like total/tax line
                if any(keyword in item_name.lower() for keyword in ['total', 'tax', 'subtotal', 'balance']):
                    continue

                item = LineItem(
                    name=item_name,
                    total_price=item_price
                )
                receipt.items.append(item)

        # Extract payment method
        payment_patterns = [
            r'(?:visa|mastercard|amex|discover|debit|credit)\s*(?:card)?',
            r'(?:cash|check)',
            r'(?:card|payment)\s*(?:ending\s*in\s*)?(\d{4})'
        ]

        for pattern in payment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                receipt.payment_method = match.group(0)
                # Extract last 4 digits if present
                last_four_match = re.search(r'(\d{4})(?:\s|$)', match.group(0))
                if last_four_match:
                    receipt.last_four = last_four_match.group(1)
                break

        return receipt

    def _parse_decimal(self, amount_str: str) -> Optional[Decimal]:
        """Parse amount string to Decimal."""
        try:
            # Remove currency symbols and commas
            cleaned = amount_str.replace('$', '').replace(',', '').strip()
            return Decimal(cleaned)
        except:
            return None

    def _categorize_receipt(self, receipt: ReceiptData) -> str:
        """
        Categorize receipt based on merchant and items.

        Categories:
        - grocery
        - restaurant
        - retail
        - gas
        - pharmacy
        - entertainment
        - other
        """
        merchant_lower = (receipt.merchant or '').lower()

        # Grocery indicators
        grocery_keywords = ['market', 'grocery', 'food', 'supermarket', 'trader joe', 'whole foods', 'safeway', 'kroger']
        if any(kw in merchant_lower for kw in grocery_keywords):
            return 'grocery'

        # Restaurant indicators
        restaurant_keywords = ['restaurant', 'cafe', 'bistro', 'grill', 'kitchen', 'bar', 'pizza', 'burger']
        if any(kw in merchant_lower for kw in restaurant_keywords):
            return 'restaurant'

        # Also check for tip (indicates restaurant)
        if receipt.tip and receipt.tip > 0:
            return 'restaurant'

        # Gas station
        gas_keywords = ['gas', 'fuel', 'shell', 'chevron', 'exxon', 'bp', 'mobil']
        if any(kw in merchant_lower for kw in gas_keywords):
            return 'gas'

        # Pharmacy
        pharmacy_keywords = ['pharmacy', 'drug', 'cvs', 'walgreens', 'rite aid']
        if any(kw in merchant_lower for kw in pharmacy_keywords):
            return 'pharmacy'

        # Retail (catch-all for stores)
        retail_keywords = ['store', 'shop', 'mart', 'amazon', 'target', 'walmart']
        if any(kw in merchant_lower for kw in retail_keywords):
            return 'retail'

        return 'other'

    def _verify_calculations(self, receipt: ReceiptData) -> bool:
        """
        Verify receipt calculations.

        Returns:
            True if calculations check out within tolerance
        """
        if not receipt.subtotal or not receipt.total:
            return False  # Can't verify without both

        # Calculate expected total
        expected_total = receipt.subtotal

        if receipt.tax:
            expected_total += receipt.tax

        if receipt.tip:
            expected_total += receipt.tip

        # Check within tolerance (1 cent)
        tolerance = Decimal('0.01')
        difference = abs(expected_total - receipt.total)

        return difference <= tolerance

    def _format_receipt_text(self, receipt: ReceiptData, raw_text: str) -> str:
        """Format receipt data for display."""
        parts = []

        if receipt.merchant:
            parts.append(f"Merchant: {receipt.merchant}")

        if receipt.date:
            parts.append(f"Date: {receipt.date}")

        if receipt.time:
            parts.append(f"Time: {receipt.time}")

        parts.append("")

        # Items
        if receipt.items:
            parts.append("Items:")
            for item in receipt.items:
                price_str = f"${item.total_price:.2f}" if item.total_price else "?"
                parts.append(f"  - {item.name}: {price_str}")

            parts.append("")

        # Totals
        if receipt.subtotal:
            parts.append(f"Subtotal: ${receipt.subtotal:.2f}")

        if receipt.tax:
            parts.append(f"Tax: ${receipt.tax:.2f}")

        if receipt.tip:
            parts.append(f"Tip: ${receipt.tip:.2f}")

        if receipt.total:
            parts.append(f"Total: ${receipt.total:.2f}")

        if receipt.payment_method:
            parts.append(f"Payment: {receipt.payment_method}")

        return '\n'.join(parts)

    def _extract_receipt_entities(self, receipt: ReceiptData) -> List[str]:
        """Extract entities from receipt."""
        entities = []

        if receipt.merchant:
            entities.append(receipt.merchant)

        if receipt.date:
            entities.append(receipt.date)

        if receipt.merchant_address:
            entities.append(receipt.merchant_address)

        # Add item names
        for item in receipt.items[:10]:  # Limit to 10
            entities.append(item.name)

        return entities

    def _extract_receipt_motifs(self, receipt: ReceiptData) -> List[str]:
        """Extract motifs from receipt."""
        motifs = ['receipt', 'purchase']

        if receipt.category:
            motifs.append(receipt.category)

        if receipt.payment_method:
            motifs.append('card_payment' if 'card' in receipt.payment_method.lower() else 'cash_payment')

        # Amount-based motifs
        if receipt.total:
            if receipt.total < Decimal('10'):
                motifs.append('small_purchase')
            elif receipt.total < Decimal('50'):
                motifs.append('medium_purchase')
            elif receipt.total < Decimal('200'):
                motifs.append('large_purchase')
            else:
                motifs.append('very_large_purchase')

        if receipt.item_count > 10:
            motifs.append('bulk_purchase')

        return motifs

    def _score_receipt_importance(
        self,
        receipt: ReceiptData,
        ocr_confidence: float
    ) -> ImportanceScore:
        """
        Score receipt importance.

        Factors:
        - Receipt completeness (all fields present)
        - Amount (higher = more important)
        - OCR confidence
        - Item count
        """
        signals = ImportanceSignals()

        # Length score (item count)
        signals.length_score = min(1.0, receipt.item_count / 10.0)

        # Technical score (not relevant for receipts)
        signals.technical_score = 0.0

        # Structural score (data completeness)
        completeness = 0.0
        if receipt.merchant:
            completeness += 0.2
        if receipt.date:
            completeness += 0.2
        if receipt.total:
            completeness += 0.3
        if receipt.items:
            completeness += 0.2
        if receipt.category:
            completeness += 0.1
        signals.structural_score = completeness

        # Authority score (OCR confidence)
        signals.authority_score = ocr_confidence

        # Engagement score (transaction size)
        if receipt.total:
            # Normalize to $200
            signals.engagement_score = min(1.0, float(receipt.total) / 200.0)

        # Custom signals
        signals.custom_signals = {
            'total_amount': float(receipt.total) if receipt.total else 0.0,
            'item_count': receipt.item_count,
            'has_category': 1.0 if receipt.category else 0.0
        }

        # Compute total
        score = signals.compute_total()

        # Generate reason
        reasons = []
        if receipt.total and receipt.total > Decimal('100'):
            reasons.append(f"${receipt.total:.2f} purchase")
        if receipt.item_count > 5:
            reasons.append(f"{receipt.item_count} items")
        if receipt.category:
            reasons.append(receipt.category)
        if completeness > 0.8:
            reasons.append("complete data")

        reason = " + ".join(reasons) if reasons else "receipt"

        return ImportanceScore(
            score=score,
            signals=signals,
            reason=reason
        )

    async def _enrich_content(self, text: str) -> tuple[List[str], List[str]]:
        """
        Extract additional entities using Ollama.

        Returns:
            (entities, motifs)
        """
        try:
            import ollama
        except ImportError:
            return [], []

        try:
            # Extract entities
            entity_prompt = f"""Extract merchant name, items, and location from this receipt.
Return only a comma-separated list.

Text: {text[:1000]}

Entities:"""

            entity_response = ollama.generate(
                model="llama3.2:3b",
                prompt=entity_prompt
            )

            entities = [
                e.strip()
                for e in entity_response['response'].split(',')
                if e.strip()
            ][:20]

            return entities, []

        except Exception as e:
            print(f"Warning: Enrichment failed: {e}")
            return [], []


# Convenience function

async def spin_receipt(
    receipt_path: str,
    verify_calculations: bool = True,
    categorize: bool = True
) -> SpinResult:
    """
    Convenience function to parse receipt.

    Args:
        receipt_path: Path to receipt image
        verify_calculations: Verify math
        categorize: Categorize receipt type

    Returns:
        SpinResult with parsed receipt data

    Example:
        >>> result = await spin_receipt("grocery_receipt.jpg")
        >>> receipt_data = result.shards[0].metadata['receipt_data']
        >>> print(f"Total: ${receipt_data['total']}")
    """
    spinner = ReceiptSpinner(
        verify_calculations=verify_calculations,
        categorize=categorize
    )

    result = await spinner.spin(receipt_path)

    return result