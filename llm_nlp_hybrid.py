"""
LLM-NLP Hybrid Constraint System
Validates and gates LLM generation through rule-based NLP constraints.

Architecture:
1. NLP stage: Parse intent, extract semantic structure, enforce constraints
2. LLM stage: Generate with NLP output as reference
3. Validation gate: LLM checks each major clause against NLP constraints before output
4. Filter: Remove zombie words, meta-commentary, unnecessary elaboration
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ConstraintLevel(Enum):
    """Severity of constraint violations"""
    BLOCK = "block"  # Don't output
    WARN = "warn"    # Flag but allow
    SOFT = "soft"    # Suggest alternative


@dataclass
class SemanticConstraint:
    """Represents a constraint on response"""
    constraint_type: str  # "scope", "length", "tone", "semantic_drift"
    description: str
    level: ConstraintLevel
    pattern: Optional[str] = None  # Regex for detection


@dataclass
class ParsedIntent:
    """Output from NLP parsing stage"""
    core_question: str
    scope_domains: List[str]  # What domains are relevant
    scope_boundaries: List[str]  # What domains are NOT relevant
    required_elements: List[str]  # Must address these
    elaboration_allowed: bool
    max_length_suggestion: int


class ZombieWordFilter:
    """Identifies and flags zombie words (AI-generated jargon)"""
    
    ZOMBIE_WORDS = {
        "fluff": ConstraintLevel.BLOCK,
        "delve into": ConstraintLevel.BLOCK,
        "exciting developments": ConstraintLevel.BLOCK,
        "in this day and age": ConstraintLevel.BLOCK,
        "it is worth noting that": ConstraintLevel.BLOCK,
        "provenance": ConstraintLevel.WARN,  # Context-dependent
        "canonical": ConstraintLevel.WARN,  # Context-dependent
        "cutting edge": ConstraintLevel.SOFT,
        "paradigm shift": ConstraintLevel.SOFT,
        "leverage": ConstraintLevel.SOFT,
        "synergy": ConstraintLevel.BLOCK,
        "utilize": ConstraintLevel.SOFT,  # Use "use" instead
    }
    
    ZOMBIE_PATTERNS = [
        r"as an (AI|language model).*(?:I|we) (cannot|can)",  # Meta-commentary
        r"(?:I|we) (?:aim|strive|endeavor) to",  # Performative
        r"it should be noted that",  # Hedging
        r"(?:furthermore|moreover|additionally),?\s+(?:it is|this is)",  # Filler
    ]
    
    @staticmethod
    def score_text(text: str) -> Tuple[ConstraintLevel, List[str]]:
        """
        Score text for zombie words.
        Returns: (max_severity_level, list_of_violations)
        """
        violations = []
        max_level = ConstraintLevel.SOFT
        
        text_lower = text.lower()
        
        # Check zombie words
        for word, level in ZombieWordFilter.ZOMBIE_WORDS.items():
            if word in text_lower:
                violations.append(f"zombie_word: {word}")
                if level == ConstraintLevel.BLOCK:
                    max_level = ConstraintLevel.BLOCK
                elif level == ConstraintLevel.WARN and max_level != ConstraintLevel.BLOCK:
                    max_level = ConstraintLevel.WARN
        
        # Check zombie patterns
        for pattern in ZombieWordFilter.ZOMBIE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"zombie_pattern: {pattern[:30]}...")
                max_level = ConstraintLevel.BLOCK
        
        return max_level, violations


class NPLParser:
    """
    NLP-stage constraint parser.
    Converts user query into structured intent with constraints.
    """
    
    def parse(self, query: str, user_context: Optional[Dict] = None) -> ParsedIntent:
        """
        Parse user query into semantic constraints.
        In production, this would use spaCy, NLTK, or custom SRL.
        """
        query_lower = query.lower()
        
        # Heuristic parsing (replace with real NLP in production)
        elaboration_allowed = any(phrase in query_lower for phrase in [
            "explain", "walk me through", "detail", "comprehensive",
            "everything", "all steps", "full breakdown"
        ])
        
        brevity_requested = any(phrase in query_lower for phrase in [
            "briefly", "in short", "tldr", "quick", "just", "minimal",
            "one sentence", "bullet points"
        ])
        
        # Extract scope boundaries
        scope_boundaries = []
        if "not" in query_lower or "don't" in query_lower or "avoid" in query_lower:
            # Simple heuristic: extract what comes after negation
            scope_boundaries = self._extract_negations(query)
        
        # Determine length suggestion
        if brevity_requested:
            max_length = 200
        elif elaboration_allowed:
            max_length = 2000
        else:
            max_length = 500
        
        return ParsedIntent(
            core_question=query,
            scope_domains=self._extract_domains(query),
            scope_boundaries=scope_boundaries,
            required_elements=self._extract_requirements(query),
            elaboration_allowed=elaboration_allowed,
            max_length_suggestion=max_length
        )
    
    @staticmethod
    def _extract_domains(query: str) -> List[str]:
        """Extract domain keywords"""
        domains = []
        domain_keywords = {
            "code": ["code", "programming", "function", "algorithm", "implement"],
            "theory": ["theory", "concept", "principle", "why", "explain"],
            "practical": ["how", "steps", "process", "do", "make"],
            "history": ["history", "origin", "when", "timeline"],
        }
        query_lower = query.lower()
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                domains.append(domain)
        return domains
    
    @staticmethod
    def _extract_negations(query: str) -> List[str]:
        """Extract what NOT to include"""
        # Simplified; real version would parse syntax tree
        negations = []
        if "not about" in query.lower():
            negations.append("off-topic elaboration")
        if "don't" in query.lower() or "don't want" in query.lower():
            negations.append("unwanted_content")
        return negations
    
    @staticmethod
    def _extract_requirements(query: str) -> List[str]:
        """Extract required response elements"""
        requirements = []
        if "step" in query.lower():
            requirements.append("sequential_steps")
        if "example" in query.lower():
            requirements.append("concrete_examples")
        if "why" in query.lower():
            requirements.append("reasoning")
        return requirements


class LLMConstraintGate:
    """
    Gate between LLM generation and output.
    LLM queries this before outputting major clauses.
    """
    
    def __init__(self):
        self.zombie_filter = ZombieWordFilter()
        self.npl_parser = NPLParser()
    
    def validate_generation(
        self,
        clause: str,
        intent: ParsedIntent,
        generation_so_far: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a clause should be output.
        Returns: (should_output, rejection_reason_or_none)
        """
        # Check zombie words
        severity, violations = self.zombie_filter.score_text(clause)
        if severity == ConstraintLevel.BLOCK:
            return False, f"Zombie word detected: {violations[0]}"
        
        # Check length constraint
        total_length = len(generation_so_far) + len(clause)
        if total_length > intent.max_length_suggestion:
            return False, "Exceeds length suggestion"
        
        # Check semantic drift: is this addressing required elements?
        if intent.required_elements:
            clause_lower = clause.lower()
            addresses_requirement = any(
                req.lower() in clause_lower or self._semantic_related(req, clause)
                for req in intent.required_elements
            )
            if not addresses_requirement and "further" not in clause_lower:
                return False, "Doesn't address required elements"
        
        # Check scope boundaries
        if intent.scope_boundaries:
            if self._violates_scope(clause, intent.scope_boundaries):
                return False, "Outside requested scope"
        
        # Check for meta-commentary
        if self._is_meta_commentary(clause):
            return False, "Meta-commentary detected"
        
        return True, None
    
    @staticmethod
    def _semantic_related(requirement: str, clause: str) -> bool:
        """Simple semantic relatedness check"""
        req_words = set(requirement.lower().split())
        clause_words = set(clause.lower().split())
        overlap = len(req_words & clause_words)
        return overlap > 0
    
    @staticmethod
    def _violates_scope(clause: str, boundaries: List[str]) -> bool:
        """Check if clause violates scope boundaries"""
        clause_lower = clause.lower()
        for boundary in boundaries:
            if boundary.lower() in clause_lower:
                return True
        return False
    
    @staticmethod
    def _is_meta_commentary(clause: str) -> bool:
        """Detect self-referential commentary"""
        meta_indicators = [
            "as an ai",
            "as a language model",
            "i should note",
            "i cannot",
            "i can only",
            "my training",
            "my limitations"
        ]
        clause_lower = clause.lower()
        return any(indicator in clause_lower for indicator in meta_indicators)


class HybridPipeline:
    """
    Full pipeline: NLP constraint → LLM generation → validation → output
    """
    
    def __init__(self):
        self.parser = NPLParser()
        self.gate = LLMConstraintGate()
    
    def process(
        self,
        query: str,
        llm_generate_fn,  # Function that generates text (mocked here)
        user_context: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """
        Process query through full pipeline.
        Returns: (final_output, metadata)
        """
        metadata = {}
        
        # Stage 1: NLP parsing
        intent = self.parser.parse(query, user_context)
        metadata["intent"] = intent
        
        # Stage 2: Generate (mocked; replace with real LLM call)
        raw_generation = llm_generate_fn(query)
        metadata["raw_generation_length"] = len(raw_generation)
        
        # Stage 3: Split into clauses and validate
        clauses = self._split_clauses(raw_generation)
        filtered_clauses = []
        rejections = []
        
        generation_so_far = ""
        for clause in clauses:
            should_output, reason = self.gate.validate_generation(
                clause, intent, generation_so_far
            )
            if should_output:
                filtered_clauses.append(clause)
                generation_so_far += clause
            else:
                rejections.append((clause[:50], reason))
        
        final_output = "".join(filtered_clauses)
        
        metadata["clauses_total"] = len(clauses)
        metadata["clauses_output"] = len(filtered_clauses)
        metadata["clauses_rejected"] = rejections
        metadata["reduction_ratio"] = len(final_output) / len(raw_generation)
        
        return final_output, metadata
    
    @staticmethod
    def _split_clauses(text: str) -> List[str]:
        """Split text into clauses for validation"""
        # Simple heuristic; real version uses proper parsing
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s + " " for s in sentences if s.strip()]


# Example usage
if __name__ == "__main__":
    pipeline = HybridPipeline()
    
    # Mock LLM generator
    def mock_llm(query: str) -> str:
        return (
            "The answer to your question is quite complex. Let me delve into this "
            "exciting development in the field. As an AI language model, I cannot "
            "provide a canonical provenance of the concept, but I can explain the "
            "robust framework. Moreover, it is worth noting that the paradigm shift "
            "has been significant. In this day and age, we leverage such systems. "
            "The key point is: X, Y, Z. Further elaboration includes more fluff."
        )
    
    query = "briefly explain X"
    output, metadata = pipeline.process(query, mock_llm)
    
    print("Original length:", metadata["raw_generation_length"])
    print("Filtered length:", len(output))
    print("Reduction ratio:", f"{metadata['reduction_ratio']:.2%}")
    print("\nFiltered output:")
    print(output)
    print("\nRejected clauses:")
    for clause, reason in metadata["clauses_rejected"]:
        print(f"  - {clause}... ({reason})")
