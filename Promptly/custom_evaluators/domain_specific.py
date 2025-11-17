#!/usr/bin/env python3
"""
Domain-Specific Evaluators

Specialized evaluators for different domains:
- Medical: Terminology accuracy, contraindications, safety
- Legal: Citation format, legal terminology, compliance
- Code: Syntax checking, security vulnerabilities, best practices
- Marketing: Brand voice consistency, tone analysis, messaging
"""

from typing import Dict, Any, Optional, List, Set, Tuple
import re
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from promptly.plugins.base import BaseEvaluator


class DomainSpecificEvaluator(BaseEvaluator):
    """
    Base class for domain-specific evaluators.

    Provides common functionality for domain validation.
    """

    def __init__(self, name: str, description: str, domain: str):
        """
        Initialize domain-specific evaluator

        Args:
            name: Evaluator name
            description: Evaluator description
            domain: Domain name (medical, legal, code, marketing)
        """
        super().__init__(name, description)
        self.domain = domain


class MedicalTerminologyEvaluator(DomainSpecificEvaluator):
    """
    Medical terminology and accuracy evaluator.

    Validates:
    - Medical terminology correctness
    - Contraindication warnings
    - Drug interaction mentions
    - Anatomical accuracy
    - Professional language
    """

    def __init__(self, terminology_db: Optional[Dict[str, List[str]]] = None):
        """
        Initialize medical terminology evaluator

        Args:
            terminology_db: Dictionary of medical terms and their synonyms/variations
        """
        super().__init__(
            name="medical_terminology",
            description="Medical terminology accuracy and safety evaluator",
            domain="medical"
        )

        # Default medical terminology database (simplified)
        self.terminology_db = terminology_db or {
            'conditions': ['diabetes', 'hypertension', 'asthma', 'copd', 'pneumonia'],
            'medications': ['aspirin', 'ibuprofen', 'acetaminophen', 'metformin', 'lisinopril'],
            'anatomy': ['heart', 'lung', 'kidney', 'liver', 'brain', 'cardiac', 'pulmonary', 'renal', 'hepatic'],
            'symptoms': ['fever', 'cough', 'pain', 'dyspnea', 'tachycardia', 'hypotension'],
            'procedures': ['surgery', 'biopsy', 'catheterization', 'intubation']
        }

        # Common medical term misspellings/errors
        self.common_errors = {
            'diabetus': 'diabetes',
            'high blood pressure': 'hypertension',
            'breathing problem': 'dyspnea',
            'heart beating fast': 'tachycardia'
        }

    def _check_medical_terminology(self, text: str) -> Dict[str, Any]:
        """Check for medical terminology usage"""
        text_lower = text.lower()

        found_terms = {
            category: [term for term in terms if term in text_lower]
            for category, terms in self.terminology_db.items()
        }

        # Check for common errors
        found_errors = {
            error: correct
            for error, correct in self.common_errors.items()
            if error in text_lower
        }

        return {
            'found_terms': found_terms,
            'term_count': sum(len(terms) for terms in found_terms.values()),
            'found_errors': found_errors,
            'error_count': len(found_errors)
        }

    def _check_contraindications(self, text: str) -> Dict[str, Any]:
        """Check for contraindication mentions"""
        contraindication_keywords = [
            'contraindication', 'contraindicated', 'should not',
            'avoid', 'do not use', 'warning', 'caution',
            'side effect', 'adverse', 'interaction'
        ]

        found_warnings = [
            kw for kw in contraindication_keywords
            if kw in text.lower()
        ]

        return {
            'has_warnings': len(found_warnings) > 0,
            'warning_keywords': found_warnings,
            'warning_count': len(found_warnings)
        }

    def _check_professional_language(self, text: str) -> Dict[str, Any]:
        """Check for professional medical language"""
        unprofessional_terms = [
            'tummy', 'pee', 'poop', 'belly', 'butt'
        ]

        professional_terms = [
            'abdomen', 'urine', 'stool', 'gluteal', 'patient',
            'physician', 'clinical', 'diagnosis', 'treatment'
        ]

        found_unprofessional = [
            term for term in unprofessional_terms
            if term in text.lower()
        ]

        found_professional = [
            term for term in professional_terms
            if term in text.lower()
        ]

        return {
            'is_professional': len(found_unprofessional) == 0,
            'unprofessional_terms': found_unprofessional,
            'professional_terms': found_professional,
            'professionalism_score': len(found_professional) / max(1, len(found_professional) + len(found_unprofessional))
        }

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate medical terminology accuracy

        Args:
            actual: Text to evaluate
            expected: Expected output
            context: Optional context

        Returns:
            float: Score between 0.0 and 1.0
        """
        if not actual:
            return 0.0

        # Check terminology
        terminology = self._check_medical_terminology(actual)

        # Check contraindications
        contraindications = self._check_contraindications(actual)

        # Check professional language
        professionalism = self._check_professional_language(actual)

        # Calculate score
        score = 0.0

        # Professional language: 40%
        score += professionalism['professionalism_score'] * 0.4

        # No terminology errors: 30%
        if terminology['error_count'] == 0:
            score += 0.3

        # Uses medical terminology: 20%
        if terminology['term_count'] > 0:
            score += 0.2

        # Mentions safety/warnings when appropriate: 10%
        if contraindications['has_warnings']:
            score += 0.1

        return min(1.0, score)

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed medical terminology metrics"""
        terminology = self._check_medical_terminology(actual)
        contraindications = self._check_contraindications(actual)
        professionalism = self._check_professional_language(actual)

        return {
            'score': self.evaluate(actual, expected, context),
            'terminology': terminology,
            'contraindications': contraindications,
            'professionalism': professionalism
        }


class LegalCitationEvaluator(DomainSpecificEvaluator):
    """
    Legal citation and terminology evaluator.

    Validates:
    - Citation format (Bluebook, etc.)
    - Legal terminology correctness
    - Case law references
    - Statute citations
    - Professional legal language
    """

    def __init__(self, citation_style: str = "bluebook"):
        """
        Initialize legal citation evaluator

        Args:
            citation_style: Citation style ('bluebook', 'alwd', 'oscola')
        """
        super().__init__(
            name=f"legal_citation_{citation_style}",
            description=f"Legal citation and terminology evaluator ({citation_style})",
            domain="legal"
        )
        self.citation_style = citation_style

        # Legal terminology
        self.legal_terms = {
            'plaintiff', 'defendant', 'appellant', 'appellee',
            'jurisdiction', 'precedent', 'statute', 'regulation',
            'motion', 'brief', 'discovery', 'deposition',
            'tort', 'contract', 'liability', 'damages'
        }

    def _check_citations(self, text: str) -> Dict[str, Any]:
        """Check for legal citations"""
        # Simplified citation patterns (Bluebook-style)
        case_citation_pattern = r'\d+\s+[A-Z][a-z.]+\s+\d+'  # e.g., "123 F.3d 456"
        statute_pattern = r'\d+\s+U\.?S\.?C\.?\s+§?\s*\d+'  # e.g., "42 USC § 1983"
        federal_reporter = r'\d+\s+F\.(2d|3d)\s+\d+'

        case_citations = re.findall(case_citation_pattern, text)
        statute_citations = re.findall(statute_pattern, text, re.IGNORECASE)
        federal_citations = re.findall(federal_reporter, text)

        return {
            'has_citations': len(case_citations) + len(statute_citations) + len(federal_citations) > 0,
            'case_citations': case_citations,
            'statute_citations': statute_citations,
            'federal_citations': federal_citations,
            'total_citations': len(case_citations) + len(statute_citations) + len(federal_citations)
        }

    def _check_legal_terminology(self, text: str) -> Dict[str, Any]:
        """Check for legal terminology usage"""
        text_lower = text.lower()

        found_terms = [
            term for term in self.legal_terms
            if term in text_lower
        ]

        return {
            'found_terms': found_terms,
            'term_count': len(found_terms),
            'uses_legal_language': len(found_terms) > 0
        }

    def _check_professional_format(self, text: str) -> Dict[str, Any]:
        """Check professional legal formatting"""
        # Check for proper structure
        has_introduction = bool(re.search(r'(introduction|background|issue presented)', text, re.IGNORECASE))
        has_analysis = bool(re.search(r'(analysis|discussion|argument)', text, re.IGNORECASE))
        has_conclusion = bool(re.search(r'(conclusion|holding|disposition)', text, re.IGNORECASE))

        return {
            'has_introduction': has_introduction,
            'has_analysis': has_analysis,
            'has_conclusion': has_conclusion,
            'is_well_structured': has_introduction and has_analysis and has_conclusion
        }

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate legal citation and terminology"""
        if not actual:
            return 0.0

        citations = self._check_citations(actual)
        terminology = self._check_legal_terminology(actual)
        formatting = self._check_professional_format(actual)

        score = 0.0

        # Has proper citations: 40%
        if citations['has_citations']:
            score += 0.4

        # Uses legal terminology: 30%
        if terminology['uses_legal_language']:
            score += 0.3

        # Well-structured: 30%
        if formatting['is_well_structured']:
            score += 0.3

        return min(1.0, score)

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed legal citation metrics"""
        citations = self._check_citations(actual)
        terminology = self._check_legal_terminology(actual)
        formatting = self._check_professional_format(actual)

        return {
            'score': self.evaluate(actual, expected, context),
            'citations': citations,
            'terminology': terminology,
            'formatting': formatting
        }


class CodeSecurityEvaluator(DomainSpecificEvaluator):
    """
    Code security and best practices evaluator.

    Validates:
    - Security vulnerabilities (SQL injection, XSS, etc.)
    - Code quality and best practices
    - Syntax correctness
    - Documentation
    - Error handling
    """

    def __init__(self, language: str = "python"):
        """
        Initialize code security evaluator

        Args:
            language: Programming language ('python', 'javascript', 'java', etc.)
        """
        super().__init__(
            name=f"code_security_{language}",
            description=f"Code security and best practices evaluator ({language})",
            domain="code"
        )
        self.language = language

        # Security vulnerability patterns
        self.vulnerability_patterns = {
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%s.*["\']\s*%',
                r'execute\s*\(\s*f["\'].*\{.*\}.*["\']\s*\)',
                r'cursor\.execute\s*\(\s*.*\+.*\)'
            ],
            'xss': [
                r'innerHTML\s*=',
                r'document\.write\s*\(',
                r'eval\s*\('
            ],
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']'
            ],
            'insecure_random': [
                r'random\.random\(\)',
                r'Math\.random\(\)'
            ]
        }

    def _check_security_vulnerabilities(self, code: str) -> Dict[str, Any]:
        """Check for security vulnerabilities"""
        found_vulnerabilities = {}

        for vuln_type, patterns in self.vulnerability_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, code)
                matches.extend(found)

            if matches:
                found_vulnerabilities[vuln_type] = matches

        return {
            'has_vulnerabilities': len(found_vulnerabilities) > 0,
            'vulnerabilities': found_vulnerabilities,
            'vulnerability_count': len(found_vulnerabilities)
        }

    def _check_best_practices(self, code: str) -> Dict[str, Any]:
        """Check coding best practices"""
        # Check for documentation
        has_docstrings = bool(re.search(r'(""".*?"""|\'\'\'.*?\'\'\')', code, re.DOTALL))
        has_comments = bool(re.search(r'#.*', code))

        # Check for error handling
        has_try_except = bool(re.search(r'try:', code))

        # Check for type hints (Python)
        has_type_hints = bool(re.search(r'->', code)) if self.language == 'python' else None

        # Check for long functions (over 50 lines)
        function_pattern = r'def\s+\w+\s*\([^)]*\):.*?(?=\ndef\s|\nclass\s|\Z)'
        functions = re.findall(function_pattern, code, re.DOTALL)
        long_functions = [f for f in functions if len(f.split('\n')) > 50]

        return {
            'has_documentation': has_docstrings or has_comments,
            'has_error_handling': has_try_except,
            'has_type_hints': has_type_hints,
            'has_long_functions': len(long_functions) > 0,
            'long_function_count': len(long_functions)
        }

    def _check_syntax(self, code: str) -> Dict[str, Any]:
        """Check syntax correctness"""
        if self.language == 'python':
            try:
                compile(code, '<string>', 'exec')
                return {'is_valid_syntax': True, 'errors': []}
            except SyntaxError as e:
                return {
                    'is_valid_syntax': False,
                    'errors': [str(e)]
                }
        else:
            # For other languages, do basic checks
            # Check for balanced braces/brackets/parentheses
            balance = {'{': 0, '[': 0, '(': 0}
            for char in code:
                if char in balance:
                    balance[char] += 1
                elif char == '}':
                    balance['{'] -= 1
                elif char == ']':
                    balance['['] -= 1
                elif char == ')':
                    balance['('] -= 1

            is_balanced = all(count == 0 for count in balance.values())

            return {
                'is_valid_syntax': is_balanced,
                'errors': [] if is_balanced else ['Unbalanced brackets/braces/parentheses']
            }

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate code security and quality"""
        if not actual:
            return 0.0

        security = self._check_security_vulnerabilities(actual)
        best_practices = self._check_best_practices(actual)
        syntax = self._check_syntax(actual)

        score = 0.0

        # Valid syntax: 30%
        if syntax['is_valid_syntax']:
            score += 0.3

        # No security vulnerabilities: 40%
        if not security['has_vulnerabilities']:
            score += 0.4
        else:
            # Partial credit if only minor issues
            penalty = min(0.4, security['vulnerability_count'] * 0.1)
            score += max(0.0, 0.4 - penalty)

        # Follows best practices: 30%
        best_practice_score = 0.0
        if best_practices['has_documentation']:
            best_practice_score += 0.1
        if best_practices['has_error_handling']:
            best_practice_score += 0.1
        if not best_practices['has_long_functions']:
            best_practice_score += 0.1

        score += best_practice_score

        return min(1.0, score)

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed code security metrics"""
        security = self._check_security_vulnerabilities(actual)
        best_practices = self._check_best_practices(actual)
        syntax = self._check_syntax(actual)

        return {
            'score': self.evaluate(actual, expected, context),
            'security': security,
            'best_practices': best_practices,
            'syntax': syntax,
            'language': self.language
        }


class BrandVoiceEvaluator(DomainSpecificEvaluator):
    """
    Brand voice consistency and tone evaluator.

    Validates:
    - Brand voice consistency
    - Tone appropriateness
    - Messaging alignment
    - Key phrase usage
    - Style guidelines compliance
    """

    def __init__(self, brand_guidelines: Optional[Dict[str, Any]] = None):
        """
        Initialize brand voice evaluator

        Args:
            brand_guidelines: Dictionary with brand guidelines
        """
        super().__init__(
            name="brand_voice",
            description="Brand voice consistency and tone evaluator",
            domain="marketing"
        )

        self.brand_guidelines = brand_guidelines or {
            'voice_attributes': ['friendly', 'professional', 'helpful'],
            'key_phrases': ['we', 'our customers', 'together', 'support'],
            'avoid_phrases': ['cheap', 'expensive', 'complicated'],
            'tone': 'conversational',
            'formality': 'semi-formal'
        }

    def _check_voice_attributes(self, text: str) -> Dict[str, Any]:
        """Check for brand voice attributes"""
        voice_indicators = {
            'friendly': ['happy', 'excited', 'glad', 'pleased', 'welcome', 'thanks'],
            'professional': ['professional', 'expert', 'qualified', 'experienced'],
            'helpful': ['help', 'assist', 'support', 'guide', 'show you', 'walk you through'],
            'casual': ['hey', 'guys', 'awesome', 'cool', 'yeah'],
            'formal': ['furthermore', 'moreover', 'therefore', 'hereby', 'pursuant']
        }

        text_lower = text.lower()
        found_attributes = {}

        for attribute, indicators in voice_indicators.items():
            found = [ind for ind in indicators if ind in text_lower]
            if found:
                found_attributes[attribute] = found

        return {
            'found_attributes': found_attributes,
            'matches_voice': any(
                attr in found_attributes
                for attr in self.brand_guidelines.get('voice_attributes', [])
            )
        }

    def _check_key_phrases(self, text: str) -> Dict[str, Any]:
        """Check for required and forbidden phrases"""
        text_lower = text.lower()

        key_phrases = self.brand_guidelines.get('key_phrases', [])
        avoid_phrases = self.brand_guidelines.get('avoid_phrases', [])

        found_key_phrases = [
            phrase for phrase in key_phrases
            if phrase.lower() in text_lower
        ]

        found_avoid_phrases = [
            phrase for phrase in avoid_phrases
            if phrase.lower() in text_lower
        ]

        return {
            'found_key_phrases': found_key_phrases,
            'key_phrase_count': len(found_key_phrases),
            'found_avoid_phrases': found_avoid_phrases,
            'has_forbidden_phrases': len(found_avoid_phrases) > 0
        }

    def _check_tone(self, text: str) -> Dict[str, Any]:
        """Check tone appropriateness"""
        # Count exclamation marks (enthusiasm)
        exclamations = text.count('!')

        # Count question marks (engagement)
        questions = text.count('?')

        # Average sentence length (formality indicator)
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        avg_sentence_length = sum(sentence_lengths) / max(1, len(sentence_lengths))

        # Determine tone
        is_enthusiastic = exclamations > len(sentences) * 0.3
        is_engaging = questions > 0
        is_formal = avg_sentence_length > 20

        return {
            'exclamations': exclamations,
            'questions': questions,
            'avg_sentence_length': avg_sentence_length,
            'is_enthusiastic': is_enthusiastic,
            'is_engaging': is_engaging,
            'is_formal': is_formal,
            'tone_score': self._calculate_tone_score(is_enthusiastic, is_engaging, is_formal)
        }

    def _calculate_tone_score(self, enthusiastic: bool, engaging: bool, formal: bool) -> float:
        """Calculate tone alignment score"""
        target_tone = self.brand_guidelines.get('tone', 'conversational')

        if target_tone == 'conversational':
            return (0.4 if engaging else 0.0) + (0.3 if not formal else 0.0) + (0.3 if enthusiastic else 0.0)
        elif target_tone == 'professional':
            return (0.5 if formal else 0.0) + (0.5 if not enthusiastic else 0.0)
        elif target_tone == 'enthusiastic':
            return (0.6 if enthusiastic else 0.0) + (0.4 if engaging else 0.0)
        else:
            return 0.5

    def evaluate(self, actual: str, expected: str,
                context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate brand voice consistency"""
        if not actual:
            return 0.0

        voice = self._check_voice_attributes(actual)
        phrases = self._check_key_phrases(actual)
        tone = self._check_tone(actual)

        score = 0.0

        # Voice attributes: 30%
        if voice['matches_voice']:
            score += 0.3

        # Key phrases: 20%
        if phrases['key_phrase_count'] > 0:
            score += 0.2

        # No forbidden phrases: 20%
        if not phrases['has_forbidden_phrases']:
            score += 0.2

        # Tone alignment: 30%
        score += tone['tone_score'] * 0.3

        return min(1.0, score)

    def get_metrics(self, actual: str, expected: str,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed brand voice metrics"""
        voice = self._check_voice_attributes(actual)
        phrases = self._check_key_phrases(actual)
        tone = self._check_tone(actual)

        return {
            'score': self.evaluate(actual, expected, context),
            'voice': voice,
            'phrases': phrases,
            'tone': tone,
            'brand_guidelines': self.brand_guidelines
        }


if __name__ == '__main__':
    # Example usage
    print("=" * 80)
    print("Domain-Specific Evaluator Examples")
    print("=" * 80)

    # Example 1: Medical Terminology
    print("\n1. Medical Terminology Evaluator")
    print("-" * 80)

    med_evaluator = MedicalTerminologyEvaluator()

    professional_text = """
    The patient presents with acute dyspnea and tachycardia. Pulmonary examination
    reveals decreased breath sounds bilaterally. Cardiac enzymes are elevated,
    suggesting possible myocardial involvement. Recommend immediate cardiology
    consultation and consideration of anticoagulation therapy. Contraindications
    include active bleeding and recent surgery.
    """

    unprofessional_text = """
    The patient has a tummy ache and can't breathe good. Their heart is beating
    real fast. Maybe they need some medicine or something.
    """

    print("\nProfessional Medical Text:")
    metrics = med_evaluator.get_metrics(professional_text, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Medical terms found: {metrics['terminology']['term_count']}")
    print(f"Professional: {metrics['professionalism']['is_professional']}")

    print("\nUnprofessional Text:")
    metrics = med_evaluator.get_metrics(unprofessional_text, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Unprofessional terms: {metrics['professionalism']['unprofessional_terms']}")

    # Example 2: Legal Citations
    print("\n\n2. Legal Citation Evaluator")
    print("-" * 80)

    legal_evaluator = LegalCitationEvaluator()

    proper_legal = """
    Introduction: The plaintiff brings this motion under 42 USC § 1983.

    Analysis: The court in Miranda v. Arizona, 384 U.S. 436 (1966), established
    that suspects must be informed of their rights. The defendant failed to
    provide these warnings as required by precedent.

    Conclusion: Therefore, the evidence should be suppressed under the exclusionary rule.
    """

    improper_legal = """
    The person is suing because their rights were violated. Some court case said
    they need to be told their rights before questioning. So the evidence should
    probably be thrown out.
    """

    print("\nProper Legal Writing:")
    metrics = legal_evaluator.get_metrics(proper_legal, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Citations found: {metrics['citations']['total_citations']}")
    print(f"Well-structured: {metrics['formatting']['is_well_structured']}")

    print("\nImproper Legal Writing:")
    metrics = legal_evaluator.get_metrics(improper_legal, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Has citations: {metrics['citations']['has_citations']}")

    # Example 3: Code Security
    print("\n\n3. Code Security Evaluator")
    print("-" * 80)

    code_evaluator = CodeSecurityEvaluator(language="python")

    secure_code = '''
def get_user(user_id: int) -> dict:
    """
    Retrieve user information from database.

    Args:
        user_id: User ID to retrieve

    Returns:
        dict: User information
    """
    try:
        query = "SELECT * FROM users WHERE id = ?"
        cursor.execute(query, (user_id,))
        return cursor.fetchone()
    except Exception as e:
        logger.error(f"Error retrieving user: {e}")
        raise
'''

    insecure_code = '''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = %s" % user_id
    cursor.execute(query)
    return cursor.fetchone()
'''

    print("\nSecure Code:")
    metrics = code_evaluator.get_metrics(secure_code, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Has vulnerabilities: {metrics['security']['has_vulnerabilities']}")
    print(f"Has documentation: {metrics['best_practices']['has_documentation']}")

    print("\nInsecure Code:")
    metrics = code_evaluator.get_metrics(insecure_code, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Vulnerabilities: {list(metrics['security']['vulnerabilities'].keys())}")

    # Example 4: Brand Voice
    print("\n\n4. Brand Voice Evaluator")
    print("-" * 80)

    brand_evaluator = BrandVoiceEvaluator()

    on_brand = """
    We're excited to help you with your question! Our team is here to support you
    every step of the way. Together, we'll find the perfect solution for your needs.
    Thank you for choosing us!
    """

    off_brand = """
    This is a complicated issue that's going to be expensive to fix. We don't
    really have time for this right now. Maybe try calling someone else.
    """

    print("\nOn-Brand Content:")
    metrics = brand_evaluator.get_metrics(on_brand, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Matches voice: {metrics['voice']['matches_voice']}")
    print(f"Key phrases found: {metrics['phrases']['key_phrase_count']}")

    print("\nOff-Brand Content:")
    metrics = brand_evaluator.get_metrics(off_brand, "")
    print(f"Score: {metrics['score']:.2f}")
    print(f"Forbidden phrases: {metrics['phrases']['found_avoid_phrases']}")

    print("\n" + "=" * 80)
