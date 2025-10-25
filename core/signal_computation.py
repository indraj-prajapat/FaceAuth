# core/signal_computation.py (updated)
import numpy as np

class SignalComputer:
    def __init__(self):
        pass

    def compute_all_signals(self, search_results, morph_prob, cohort_count, forns):
        signals = {}
        sim, individual = self.compute_similarity(search_results)
        signals['sim'] = sim
        signals['individual_sims'] = individual

        agree, top_candidate = self.compute_agreement(search_results)
        signals['agree'] = agree
        signals['top_candidate'] = top_candidate

        signals['margin'] = self.compute_margin(search_results)
        signals['morph'] = morph_prob
        signals['forns'] = float(forns)
        signals['cohort'] = self.compute_cohortness(cohort_count)
        signals['uncertainty'] = self.compute_uncertainty(individual)
        return signals

    def compute_similarity(self, search_results):
        sim_arc = search_results['arcface'][0][0] if len(search_results['arcface'][0]) > 0 else 0
        sim_ada = search_results['adaface'][0][0] if len(search_results['adaface'][0]) > 0 else 0
        sim_ela = search_results['elastic'][0][0] if len(search_results['elastic'][0]) > 0 else 0
        avg = (sim_arc + sim_ada + sim_ela) / 3.0
        return avg * 100.0, [sim_arc * 100.0, sim_ada * 100.0, sim_ela * 100.0]

    def compute_agreement(self, search_results):
        t_arc = search_results['arcface'][1][0] if len(search_results['arcface'][1]) > 0 else -1
        t_ada = search_results['adaface'][1][0] if len(search_results['adaface'][1]) > 0 else -1
        t_ela = search_results['elastic'][1][0] if len(search_results['elastic'][1]) > 0 else -1
        candidates = [t_arc, t_ada, t_ela]
        if -1 in candidates:
            return 0, -1
        mc = max(set(candidates), key=candidates.count)
        cnt = candidates.count(mc)
        return (cnt / 3.0) * 100.0, mc

    def compute_margin(self, search_results):
        margins = []
        THRESH = 1e3  # similarity must be in [-1, 1], so large values are invalid
        for m in ['arcface', 'adaface', 'elastic']:
            sims = search_results[m][0]
            # print('sims    ::', sims[:3])
            if len(sims) < 2:
                continue
            # Clip similarity values to safe range
            s1 = sims[0] if abs(sims[0]) < THRESH else 0
            s2 = sims[1] if abs(sims[1]) < THRESH else 0
            if not np.isfinite(s1) or not np.isfinite(s2):
                continue
            margin_value = (s1 - s2) * 100.0
            margins.append(margin_value)
        if not margins:
            print("Warning: No valid margins found, returning 0")
            return 0
        return min(margins)



    def compute_cohortness(self, count):
        if count >= 5:
            return 100.0
        return min((count / 5) * 100.0, 100.0)

    def compute_uncertainty(self, individual_sims):
        return float(min(np.std(individual_sims) * 1000.0, 100.0))
