from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Dict, Optional


class RiskProfile(Enum):
    """Risk profiles for different use cases"""
    STRICT = "strict"       # Government/banking (high false positives acceptable)
    BALANCED = "balanced"   # General (default)
    LENIENT = "lenient"     # Social media (high false negatives acceptable)


@dataclass
class RiskWeights:
    """7-signal fusion weights; sum should be ~1.0"""
    w_sim: float = 0.30     # Similarity (100 - SIM)
    w_agree: float = 0.15   # Agreement across models
    w_margin: float = 0.22  # Gap between top-2 candidates
    w_morph: float = 0.20   # Morphing detection
    w_forns: float = 0.12   # Forensic artifacts
    w_cohort: float = 0.17  # Reuse frequency
    w_unc: float = 0.04     # Uncertainty/variance


@dataclass
class RiskThresholds:
    """Adaptive thresholds per profile"""
    low_cut: float       # Risk <= low_cut → LOW risk
    high_cut: float      # Risk > high_cut → HIGH risk
    # Between low_cut and high_cut → MEDIUM risk
    
    morph_hard: float    # Hard override threshold (%)
    cohort_hard: float   # Hard override threshold
    margin_hard: float   # Minimum margin for near-duplicate check


class RiskScorer:
    """Production-ready risk scorer with profiles, audit trail, and confidence"""
    
    PROFILES = {
        RiskProfile.STRICT: RiskThresholds(
            low_cut=20.0,
            high_cut=40.0,
            morph_hard=60.0,     # Lower threshold (stricter)
            cohort_hard=70.0,
            margin_hard=5.0
        ),
        RiskProfile.BALANCED: RiskThresholds(
            low_cut=30.0,
            high_cut=60.0,
            morph_hard=80.0,     # Default
            cohort_hard=90.0,
            margin_hard=3.0
        ),
        RiskProfile.LENIENT: RiskThresholds(
            low_cut=45.0,
            high_cut=75.0,
            morph_hard=95.0,     # Higher threshold (lenient)
            cohort_hard=95.0,
            margin_hard=2.0
        ),
    }
    
    def __init__(self, profile: RiskProfile = RiskProfile.LENIENT, weights: RiskWeights = None):
        self.profile = profile
        self.thresholds = self.PROFILES[profile]
        self.w = weights or RiskWeights()
    
    def compute_risk_score(self, signals: Dict) -> tuple:
        """
        Compute risk score and component contributions
        Returns: (risk_score, components_dict, audit_notes)
        """
        SIM = float(signals.get('sim', 0.0))
        AGREE = float(signals.get('agree', 0.0))
        MARGIN = float(signals.get('margin', 0.0))
        FORNS = float(signals.get('forns', 0.0))
        COHORT = float(signals.get('cohort', 0.0))
        UNC = float(signals.get('uncertainty', 0.0))
        
        morph_signal = signals.get('morph')
        if isinstance(morph_signal, (tuple, list)) and len(morph_signal) > 0:
            MORPH = float(morph_signal[0])
        else:
            MORPH = float(morph_signal)
        
        audit_notes = []
        components = {}
        # Policy override: low similarity => unique
        
        # 1. Similarity contribution (higher SIM = lower risk)
        sim_contrib = (100.0 - SIM) * self.w.w_sim
        components['sim_contrib'] = sim_contrib
        audit_notes.append(f"SIM={SIM:.1f}% → contrib={sim_contrib:.1f} (higher sim is better)")
        
        # 2. Agreement contribution (consensus of models)
        agree_contrib = (100.0 - AGREE) * self.w.w_agree
        components['agree_contrib'] = agree_contrib
        audit_notes.append(f"AGREE={AGREE:.1f}% → contrib={agree_contrib:.1f}")
        
        # 3. Margin penalty (LINEAR for stability; higher margin = lower risk)
        if MARGIN >= self.thresholds.margin_hard:
            margin_penalty = 0.0
            audit_notes.append(f"MARGIN={MARGIN:.1f} ≥ {self.thresholds.margin_hard} → No penalty (clear winner)")
        elif MARGIN >= 0:
            # Linear penalty: 0 margin = 100 penalty, high margin = 0 penalty
            margin_penalty = (self.thresholds.margin_hard - MARGIN) / self.thresholds.margin_hard * 100.0
            audit_notes.append(f"MARGIN={MARGIN:.1f} → linear penalty={margin_penalty:.1f}")
        else:
            margin_penalty = 100.0  # No positive margin
            audit_notes.append(f"MARGIN={MARGIN:.1f} (negative) → max penalty=100")
        
        margin_contrib = margin_penalty * self.w.w_margin
        components['margin_contrib'] = margin_contrib
        
        # 4. Morph contribution
        morph_contrib = MORPH * self.w.w_morph
        components['morph_contrib'] = morph_contrib
        audit_notes.append(f"MORPH={MORPH:.1f}% → contrib={morph_contrib:.1f}")
        
        # 5. Forensics contribution (print/rephoto/splice indicators)
        forns_contrib = FORNS * self.w.w_forns
        components['forns_contrib'] = forns_contrib
        audit_notes.append(f"FORNS={FORNS:.1f} → contrib={forns_contrib:.1f}")
        
        # 6. Cohort contribution (reuse frequency penalty)
        cohort_contrib = COHORT * self.w.w_cohort
        components['cohort_contrib'] = cohort_contrib
        audit_notes.append(f"COHORT={COHORT:.1f}% → contrib={cohort_contrib:.1f} (higher reuse = higher risk)")
        
        # 7. Uncertainty contribution
        unc_contrib = UNC * self.w.w_unc
        components['unc_contrib'] = unc_contrib
        audit_notes.append(f"UNC={UNC:.1f}% → contrib={unc_contrib:.1f}")
        
        # Total risk
        risk = (sim_contrib + agree_contrib + margin_contrib + morph_contrib + 
                forns_contrib + cohort_contrib + unc_contrib)
        risk = float(np.clip(risk, 0.0, 100.0))  # Clip to [0, 100]
        
        return risk, components, audit_notes
    
    def make_decision(self, risk: float, signals: Dict, quality_pass: bool = True,
                  morph_discrepancy: bool = False) -> Dict:
        """
        Strict policy:
        - SIM >= 90% => never 'unique'
        - 70% <= SIM < 90% => 'review'
        - SIM < 70% => can be 'unique' only if benign signals
        """

        # Extract signals
        morph_signal = signals.get('morph')
        morph_prob = float(morph_signal[0] if isinstance(morph_signal, (tuple, list)) and morph_signal else morph_signal or 0.0)
        sim = float(signals.get('sim', 0.0))           # similarity in %
        margin = float(signals.get('margin', 100.0))   # higher = clearer separation
        agree = float(signals.get('agree', 0.0))
        cohort = float(signals.get('cohort', 0.0))
        forns = float(signals.get('forns', 0.0))

        # Tunables
        SIM_STRICT_AUTO_DUP = 99.0          # very strict auto-duplicate band [policy] [web:20]
        SIM_HIGH = 90.0                     # strict boundary: never unique at/above this [web:20]
        SIM_REVIEW_LO = 50.0                # review band lower bound [web:41]
        MARGIN_SAFE = max(15.0, self.thresholds.margin_hard)    # safe rank separation [web:41]
        AGREE_STRONG = 90.0
        MORPH_SOFT = min(20.0, getattr(self.thresholds, 'morph_soft', 25.0))  # benign morph [web:29]
        FORNS_LIMIT = 30.0
        LOW_SIM_UNIQUE = 40.0  # percent
        if sim < LOW_SIM_UNIQUE and risk < 40 and morph_prob < 49.0:
            return {
                'risk_level': 'LOW',
                'status': 'unique',
                'action': 'Auto-cleared as unique (low similarity override)',
                'confidence': 0.90,
                'reason': f'SIM={sim:.1f}% < {LOW_SIM_UNIQUE:.1f}% policy threshold and {risk:.1f} risk',
                'override': 'LOW_SIM_UNIQUE'
            }
        # 1) Hard overrides
        if not quality_pass:
            return {'risk_level': 'REJECT','status': 'reject','action': 'Request re-upload with better lighting/angle',
                    'confidence': 1.0,'reason': 'Quality gate failed','override': 'QUALITY_FAILURE'}  # [web:43]

        if morph_prob > self.thresholds.morph_hard:
            return {'risk_level': 'HIGH','status': 'quarantine','action': 'Quarantine for forensic review: Possible morphed face',
                    'confidence': 0.95,'reason': f'MORPH={morph_prob:.1f}% > threshold={self.thresholds.morph_hard}%','override': 'MORPH_HARD'}  # [web:29]

        if cohort > self.thresholds.cohort_hard:
            return {'risk_level': 'HIGH','status': 'quarantine','action': 'Quarantine: Image reused excessively in database',
                    'confidence': 0.90,'reason': f'COHORT={cohort:.1f}% > threshold={self.thresholds.cohort_hard}%','override': 'COHORT_HARD'}  # [web:28]

        # Near-duplicate conflict: high sim but ambiguous margin
        if sim >= SIM_HIGH and margin < self.thresholds.margin_hard and agree > 80.0:
            return {'risk_level': 'HIGH','status': 'quarantine','action': 'Quarantine: Near-identical face with ambiguous ranking',
                    'confidence': 0.85,'reason': f'SIM={sim:.1f}% >= {SIM_HIGH}, MARGIN={margin:.1f} < {self.thresholds.margin_hard}, AGREE={agree:.1f}% > 80',
                    'override': 'NEAR_TIE'}  # [web:41]

        if morph_discrepancy:
            return {'risk_level': 'HIGH','status': 'quarantine','action': 'Quarantine: Enhancement significantly altered morph signal',
                    'confidence': 0.80,'reason': 'Enhancement-induced morph shift detected','override': 'MORPH_DISCREPANCY'}  # [web:41]

        # 2) Strict similarity policy
        # 2a) SIM >= 99%: auto-duplicate if corroborated; else review/quarantine
        if sim >= SIM_STRICT_AUTO_DUP:
            if (agree >= AGREE_STRONG and margin >= MARGIN_SAFE and morph_prob < MORPH_SOFT and forns < FORNS_LIMIT):
                return {'risk_level': 'LOW','status': 'duplicate','action': 'Auto-confirmed duplicate (strict band)',
                        'confidence': 0.995,'reason': f'SIM={sim:.2f}% >= {SIM_STRICT_AUTO_DUP}, corroborated: AGREE={agree:.1f}%, MARGIN={margin:.1f}, MORPH={morph_prob:.1f}%, FORNS={forns:.1f}%'}  # [web:20]
            # high sim but contradictions
            return {'risk_level': 'MEDIUM','status': 'review','action': 'Manual review: Very high similarity but weak corroboration',
                    'confidence': 0.75,'reason': 'SIM in strict band; corroboration insufficient (AGREE/MARGIN/MORPH/FORNS)'}  # [web:41]

        # 2b) 90% <= SIM < 99%: never unique
        if sim >= SIM_HIGH:
            if (agree >= AGREE_STRONG and margin >= MARGIN_SAFE and morph_prob < MORPH_SOFT and forns < FORNS_LIMIT):
                return {'risk_level': 'LOW','status': 'duplicate','action': 'Confirmed duplicate (high similarity with corroboration)',
                        'confidence': 0.97,'reason': f'SIM={sim:.1f}% >= {SIM_HIGH} with strong corroboration'}  # [web:20]
            # ambiguous -> review; if margin notably small, quarantine
            if margin < MARGIN_SAFE or agree < 80.0 or forns >= FORNS_LIMIT or morph_prob >= MORPH_SOFT:
                return {'risk_level': 'HIGH','status': 'quarantine','action': 'Quarantine: High similarity with ambiguous or risky signals',
                        'confidence': 0.85,'reason': f'Ambiguity/risk with SIM={sim:.1f}% (MARGIN/AGREE/MORPH/FORNS)'}  # [web:41]
            return {'risk_level': 'MEDIUM','status': 'review','action': 'Manual review: High similarity requires validation',
                    'confidence': 0.72,'reason': f'SIM={sim:.1f}% high but corroboration not decisive'}  # [web:41]

        # 2c) 70% <= SIM < 90%: always review
        if sim >= SIM_REVIEW_LO:
            return {'risk_level': 'MEDIUM','status': 'review','action': 'Send to manual review (mid similarity band)',
                    'confidence': 0.70,'reason': f'SIM={sim:.1f}% in review band [{SIM_REVIEW_LO}, {SIM_HIGH})'}  # [web:41]

        # 2d) SIM < 70%: can be unique if benign
        if sim < SIM_REVIEW_LO:
            if risk <= self.thresholds.low_cut and morph_prob < MORPH_SOFT and forns < FORNS_LIMIT:
                return {'risk_level': 'LOW','status': 'unique','action': 'Auto-cleared as unique identity',
                        'confidence': 0.90,'reason': f'Low similarity and benign signals; Risk={risk:.1f} ≤ low_cut={self.thresholds.low_cut}'}  # [web:22]
            return {'risk_level': 'MEDIUM','status': 'review','action': 'Manual review due to residual risk/signals',
                    'confidence': 0.65,'reason': 'Low similarity but not sufficiently benign for auto-unique'}  # [web:41]

        # Fallback (should not hit)
        return {'risk_level': 'MEDIUM','status': 'review','action': 'Manual review (fallback)',
                'confidence': 0.6,'reason': 'Default fallback'}  # [web:41]
