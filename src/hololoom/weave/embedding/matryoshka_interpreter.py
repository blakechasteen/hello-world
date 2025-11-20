#!/usr/bin/env python3
"""
🔥 Matryoshka Layer Interpreter
================================
Visualize and interpret what each embedding dimension layer captures.

The Matryoshka embedding architecture uses progressive dimensionality:
- 96d:  Surface features (basic keywords, entities)
- 192d: Symbolic patterns (relationships, context)
- 384d: Archetypal structures (deep meaning, universal patterns)

This interpreter shows what each layer "sees" in the narrative.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.embedding.spectral import MatryoshkaEmbeddings


class MatryoshkaInterpreter:
    """Interpret what each Matryoshka layer captures."""
    
    def __init__(self):
        """Initialize with standard scales."""
        self.scales = [96, 192, 384]
        self.embedder = MatryoshkaEmbeddings(sizes=self.scales)
        
        # Layer characteristics
        self.layer_info = {
            96: {
                'name': '🔥 Surface Layer',
                'captures': [
                    'Keywords and entities',
                    'Literal meaning',
                    'Basic sentiment',
                    'Direct references',
                    'Concrete objects'
                ],
                'depth': 'SURFACE',
                'color': '#FFD700'
            },
            192: {
                'name': '🌊 Symbolic Layer',
                'captures': [
                    'Relationships between concepts',
                    'Context and nuance',
                    'Implicit meanings',
                    'Metaphorical connections',
                    'Emotional undertones'
                ],
                'depth': 'SYMBOLIC',
                'color': '#FFA500'
            },
            384: {
                'name': '⭐ Archetypal Layer',
                'captures': [
                    'Universal patterns',
                    'Deep psychological structures',
                    'Mythological resonance',
                    'Transformation arcs',
                    'Cosmic truths'
                ],
                'depth': 'ARCHETYPAL/MYTHIC/COSMIC',
                'color': '#FF4500'
            }
        }
    
    def interpret_text(self, text: str) -> Dict[str, any]:
        """
        Interpret text across all Matryoshka layers.
        
        Returns information about what each layer captures.
        """
        # Generate embeddings
        embeddings = self.embedder.encode(text)
        
        results = {
            'text': text,
            'layers': {}
        }
        
        for scale, embedding in zip(self.scales, embeddings):
            # Analyze embedding characteristics
            magnitude = float(np.linalg.norm(embedding))
            mean_activation = float(np.mean(embedding))
            std_activation = float(np.std(embedding))
            sparsity = float(np.sum(np.abs(embedding) < 0.01) / len(embedding))
            
            # Top activations (strongest features)
            top_indices = np.argsort(np.abs(embedding))[-10:]
            top_values = embedding[top_indices]
            
            results['layers'][scale] = {
                'info': self.layer_info[scale],
                'dimensions': scale,
                'magnitude': magnitude,
                'mean_activation': mean_activation,
                'std_activation': std_activation,
                'sparsity': sparsity,
                'top_features': {
                    'indices': top_indices.tolist(),
                    'values': top_values.tolist()
                },
                'interpretation': self._interpret_layer(
                    scale, magnitude, mean_activation, std_activation, sparsity, text
                )
            }
        
        return results
    
    def _interpret_layer(
        self,
        scale: int,
        magnitude: float,
        mean: float,
        std: float,
        sparsity: float,
        text: str
    ) -> Dict[str, any]:
        """Interpret what a specific layer detected."""
        
        interpretation = {
            'strength': self._assess_strength(magnitude),
            'complexity': self._assess_complexity(std, sparsity),
            'detected_features': []
        }
        
        # Layer-specific interpretations
        if scale == 96:
            # Surface layer - look for keywords
            interpretation['detected_features'] = self._detect_surface_features(text)
            interpretation['summary'] = (
                f"Surface scan detected {len(interpretation['detected_features'])} "
                f"concrete elements with {interpretation['strength']} signal strength."
            )
            
        elif scale == 192:
            # Symbolic layer - look for patterns
            interpretation['detected_features'] = self._detect_symbolic_features(text)
            interpretation['summary'] = (
                f"Symbolic analysis found {len(interpretation['detected_features'])} "
                f"relational patterns with {interpretation['complexity']} structure."
            )
            
        elif scale == 384:
            # Archetypal layer - look for deep patterns
            interpretation['detected_features'] = self._detect_archetypal_features(text)
            interpretation['summary'] = (
                f"Archetypal resonance captured {len(interpretation['detected_features'])} "
                f"universal patterns with {interpretation['strength']} mythic depth."
            )
        
        return interpretation
    
    def _assess_strength(self, magnitude: float) -> str:
        """Assess signal strength from magnitude."""
        if magnitude > 15:
            return "INTENSE"
        elif magnitude > 10:
            return "STRONG"
        elif magnitude > 5:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _assess_complexity(self, std: float, sparsity: float) -> str:
        """Assess complexity from std dev and sparsity."""
        if std > 0.15 and sparsity < 0.3:
            return "HIGHLY COMPLEX"
        elif std > 0.10:
            return "COMPLEX"
        elif std > 0.05:
            return "MODERATE"
        else:
            return "SIMPLE"
    
    def _detect_surface_features(self, text: str) -> List[str]:
        """Detect surface-level features."""
        features = []
        
        # Keywords
        keywords = ['hero', 'journey', 'battle', 'wisdom', 'death', 'rebirth',
                   'love', 'fear', 'power', 'truth', 'darkness', 'light']
        for keyword in keywords:
            if keyword.lower() in text.lower():
                features.append(f"Keyword: '{keyword}'")
        
        # Entities (simple detection)
        words = text.split()
        capitalized = [w for w in words if w and w[0].isupper() and len(w) > 1]
        if capitalized:
            features.append(f"Entities: {', '.join(capitalized[:3])}")
        
        # Sentiment markers
        positive = ['hope', 'victory', 'triumph', 'joy', 'success']
        negative = ['fear', 'failure', 'loss', 'pain', 'defeat']
        
        if any(p in text.lower() for p in positive):
            features.append("Positive sentiment detected")
        if any(n in text.lower() for n in negative):
            features.append("Negative sentiment detected")
        
        return features
    
    def _detect_symbolic_features(self, text: str) -> List[str]:
        """Detect symbolic patterns."""
        features = []
        
        # Transformation language
        transforms = ['became', 'transformed', 'changed', 'evolved', 'discovered']
        if any(t in text.lower() for t in transforms):
            features.append("Transformation arc detected")
        
        # Relationships
        relations = ['with', 'against', 'for', 'from', 'to']
        relation_count = sum(1 for r in relations if r in text.lower())
        if relation_count > 2:
            features.append(f"Complex relationships ({relation_count} connections)")
        
        # Contrast/duality
        dualities = [
            ('light', 'dark'), ('good', 'evil'), ('life', 'death'),
            ('old', 'new'), ('weak', 'strong')
        ]
        for pair in dualities:
            if pair[0] in text.lower() and pair[1] in text.lower():
                features.append(f"Duality: {pair[0]}/{pair[1]}")
        
        # Metaphorical language
        metaphors = ['like', 'as if', 'seemed', 'appeared', 'resembled']
        if any(m in text.lower() for m in metaphors):
            features.append("Metaphorical language present")
        
        return features
    
    def _detect_archetypal_features(self, text: str) -> List[str]:
        """Detect archetypal patterns."""
        features = []
        
        # Hero's Journey stages
        journey_markers = {
            'Call to Adventure': ['called', 'summoned', 'must'],
            'Refusal': ['refused', 'denied', 'rejected'],
            'Mentor': ['guide', 'teacher', 'wisdom', 'mentor'],
            'Threshold': ['crossed', 'entered', 'left behind'],
            'Ordeal': ['faced', 'confronted', 'battle', 'test'],
            'Revelation': ['realized', 'understood', 'discovered', 'revealed'],
            'Transformation': ['changed', 'became', 'transformed', 'reborn'],
            'Return': ['returned', 'brought back', 'shared']
        }
        
        for stage, markers in journey_markers.items():
            if any(marker in text.lower() for marker in markers):
                features.append(f"Journey stage: {stage}")
        
        # Archetypal figures
        archetypes = {
            'Hero': ['hero', 'protagonist', 'warrior', 'champion'],
            'Mentor': ['mentor', 'guide', 'teacher', 'wise'],
            'Shadow': ['enemy', 'villain', 'dark', 'shadow'],
            'Trickster': ['trickster', 'fool', 'deceiver'],
            'Threshold Guardian': ['guardian', 'gatekeeper', 'obstacle']
        }
        
        for archetype, markers in archetypes.items():
            if any(marker in text.lower() for marker in markers):
                features.append(f"Archetype: {archetype}")
        
        # Universal themes
        themes = {
            'Death & Rebirth': ['death', 'rebirth', 'resurrection'],
            'Sacred Marriage': ['union', 'marriage', 'joined'],
            'Divine Child': ['child', 'innocent', 'pure'],
            'Sacrifice': ['sacrifice', 'gave', 'offered'],
            'Quest': ['quest', 'search', 'seeking']
        }
        
        for theme, markers in themes.items():
            if any(marker in text.lower() for marker in markers):
                features.append(f"Universal theme: {theme}")
        
        return features
    
    def visualize_interpretation(self, results: Dict) -> str:
        """Generate human-readable visualization."""
        output = []
        
        output.append("\n" + "="*70)
        output.append("🔥 MATRYOSHKA LAYER INTERPRETATION")
        output.append("="*70 + "\n")
        
        output.append(f"📝 Text: {results['text'][:100]}...")
        output.append("")
        
        for scale in self.scales:
            layer = results['layers'][scale]
            info = layer['info']
            interp = layer['interpretation']
            
            output.append("-" * 70)
            output.append(f"{info['name']} ({scale}d)")
            output.append("-" * 70)
            
            output.append(f"\n🎯 What this layer captures:")
            for capture in info['captures']:
                output.append(f"   • {capture}")
            
            output.append(f"\n📊 Signal Characteristics:")
            output.append(f"   • Magnitude: {layer['magnitude']:.2f}")
            output.append(f"   • Activation: {layer['mean_activation']:.4f} ± {layer['std_activation']:.4f}")
            output.append(f"   • Sparsity: {layer['sparsity']:.1%}")
            output.append(f"   • Strength: {interp['strength']}")
            output.append(f"   • Complexity: {interp['complexity']}")
            
            output.append(f"\n🔍 Detected Features:")
            if interp['detected_features']:
                for feature in interp['detected_features']:
                    output.append(f"   ✓ {feature}")
            else:
                output.append("   (No specific features detected)")
            
            output.append(f"\n💡 Summary: {interp['summary']}")
            output.append("")
        
        output.append("="*70)
        output.append("🔥 INTERPRETATION COMPLETE")
        output.append("="*70 + "\n")
        
        return "\n".join(output)


def demonstrate_interpreter():
    """Demonstrate the Matryoshka interpreter."""
    
    print("\n" + "="*70)
    print("🔥 MATRYOSHKA LAYER INTERPRETER DEMO")
    print("="*70 + "\n")
    
    interpreter = MatryoshkaInterpreter()
    
    # Test narratives
    narratives = [
        "Odysseus faced the Cyclops and overcame his pride. The journey home transformed him from warrior to wise king.",
        "The startup pivoted three times before finding product-market fit.",
        "In therapy, I finally faced the shadow I'd been avoiding. The wound became a doorway to wholeness."
    ]
    
    for i, text in enumerate(narratives, 1):
        print(f"\n📖 NARRATIVE {i}:")
        print(f"   {text}\n")
        
        results = interpreter.interpret_text(text)
        visualization = interpreter.visualize_interpretation(results)
        print(visualization)
        
        if i < len(narratives):
            print("\n" + "~"*70 + "\n")


if __name__ == "__main__":
    demonstrate_interpreter()
