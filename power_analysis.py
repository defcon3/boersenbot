#!/usr/bin/env python3
"""
POWER-ANALYSE fuer knapp-unter-Schwelle Tests auf veitluther.de/done.

Ausloeser: Reviewer-Feedback (Grok/ChatGPT) hat zwei methodische Punkte
angemerkt:
  1. Knapp-unter-Schwelle-Tests verdienen eine Power-Analyse, sonst kann
     "nicht signifikant" durch underpowered design erklaert werden.
  2. Familie-Korrektur fuer abhaengige Tests (separat).

Diese Datei adressiert Punkt 1.

METHODIK: One-sample t-Test gegen Null (Excess-Returns), einseitig
positiv (Hypothese ist gerichtet).

  Power(observed_effect) = 1 - beta = Prob(Verwerfen H0 | H1 = beobachtet)
                        = P(Z > z_alpha - mu/SE)
                        = Phi(mu/SE - z_alpha)

  MDE(power) = (z_alpha + z_beta) * sigma / sqrt(n)

mit SE = sigma / sqrt(n).

Drei Faelle aus der Done-Seite:
  A) TECH N=6 (Streak-Sektor):    n=21,  diff=+0.302%, t=+1.92 (naive 1.96)
  B) FRED H3 SPY-20d:              n=?,   diff=?,       t=-2.37 (Bonf. 3.0)
  C) CC-MR Excess minKorb>=8:      repraesentativ aus Streak-Memory

Fuer B/C extrahieren wir n und sigma aus dem t-Wert und der Differenz
(falls vorhanden), oder geben qualitative Schaetzung.
"""
import numpy as np
from scipy import stats

ALPHA = 0.05
TARGET_POWER = 0.80
Z_ALPHA_1SIDED = stats.norm.ppf(1 - ALPHA)        # 1.6449
Z_POWER = stats.norm.ppf(TARGET_POWER)             # 0.8416


def power_given_effect(n, diff, sigma, alpha=ALPHA, one_sided=True):
    """Power in Richtung des beobachteten Effekts (|diff| gegen H0=0)."""
    z_alpha = stats.norm.ppf(1 - alpha) if one_sided else stats.norm.ppf(1 - alpha/2)
    se = sigma / np.sqrt(n)
    return float(stats.norm.cdf(abs(diff) / se - z_alpha))


def mde(n, sigma, alpha=ALPHA, power=TARGET_POWER, one_sided=True):
    """Minimal detektierbare Effekt-Groesse fuer Power = 0.80."""
    z_alpha = stats.norm.ppf(1 - alpha) if one_sided else stats.norm.ppf(1 - alpha/2)
    z_power = stats.norm.ppf(power)
    return float((z_alpha + z_power) * sigma / np.sqrt(n))


def sigma_from_t(diff, t, n):
    """sigma aus beobachtetem t-Wert rekonstruieren: t = diff / (sigma/sqrt(n))."""
    return float(diff * np.sqrt(n) / t)


def report(name, n, diff, t_obs, bonferroni_t=None, naive_t=1.96):
    sigma = sigma_from_t(diff, t_obs, n)
    pow_obs = power_given_effect(n, diff, sigma)
    mde_80 = mde(n, sigma)
    n_needed_80 = int(np.ceil((Z_ALPHA_1SIDED + Z_POWER)**2 * sigma**2 / max(diff**2, 1e-12)))

    print(f"\n--- {name} ---")
    print(f"  Beobachtet:   n={n}, diff={diff*100:+.3f}%, t={t_obs:+.2f}")
    print(f"  sigma (impl): {sigma*100:.3f}%")
    print(f"  SE:           {(sigma/np.sqrt(n))*100:.3f}%")
    print(f"  Power(beob.): {pow_obs*100:.1f}%  (Standard 80%, knapp wenn <70%)")
    print(f"  MDE @80%-Power: {mde_80*100:+.3f}%   (Effekt, der bei n={n} sichtbar geworden waere)")
    print(f"  n fuer 80% Power bei beob. Effekt: {n_needed_80} Tage")
    if bonferroni_t is not None:
        ratio = abs(t_obs) / bonferroni_t
        print(f"  Bonferroni-Schwelle: {bonferroni_t:+.2f}   (t/Schwelle = {ratio:.2f})")
    print(f"  Naive-Schwelle: {naive_t}   (verfehlt um {abs(t_obs)-naive_t:+.2f})")


print("="*72)
print("POWER-ANALYSE fuer knapp-unter-Schwelle-Tests")
print("="*72)
print(f"alpha={ALPHA} (einseitig), Power-Ziel={TARGET_POWER}")
print(f"z_alpha={Z_ALPHA_1SIDED:.4f}, z_power={Z_POWER:.4f}")

# A) TECH N=6 - aus Streak-Sektor-Test
report("A) TECH N=6 (Streak-MR Sektor-Differenzierung)",
       n=21, diff=0.00302, t_obs=1.92,
       bonferroni_t=2.7, naive_t=1.96)

# B) FRED H3 SPY-20d - aus FRED V3 Test
# Werte aus Memory fred_macro_falsified.md: SPY-20d t=-2.37 (Bonferroni FAIL)
# diff_test aus Logs: ~-0.71% (kann je nach Fenster variieren)
# Bei H3 Claims-MA-Change war n_test_sig typisch 60-90 (Q75-Signaltage)
# Wir nehmen die einzeln-publizierten Zahlen: n~70, diff aequivalent
# Hier negativer Effekt - wir nehmen abs(diff) und drehen die Richtung um
report("B) FRED H3 SPY-20d (Claims-MA-Change, einseitig negativ)",
       n=70, diff=-0.0071, t_obs=-2.37,
       bonferroni_t=-3.0, naive_t=-1.96)

# C) CC-MR Excess minKorb>=8 - repraesentative Zelle aus Excess-Test
# Memory cc_meanrev_excess: "alle OOS-t < 1.6" bei minKorb>=8
# Beispiel: N=4, MR-Korb, ~250-350 Test-Tage, t typisch +1.0 bis +1.5
# Wir nehmen: n=300, t=+1.4, diff implizit kleine positive Zahl
report("C) CC-MR Excess minKorb>=8 (repraesentativ)",
       n=300, diff=0.0008, t_obs=1.40,
       bonferroni_t=None, naive_t=1.96)

print("\n" + "="*72)
print("INTERPRETATIONS-HINWEIS")
print("="*72)
print("""
Power = 80% ist die Standard-Konvention. Power < 50% heisst: die Studie
hatte mehr Wahrscheinlichkeit, den ECHTEN Effekt zu uebersehen als ihn zu
finden. "Nicht signifikant" sagt in dem Fall nichts.

Wenn MDE >> beobachteter Effekt: das Setup war underpowered fuer DIESE
Effektgroesse - eine groessere Stichprobe haette potentiell Signifikanz
gebracht (aber auch nicht zwingend, weil der "beobachtete Effekt" selbst
Rauschen sein kann).

WICHTIG: Power-Analyse rettet kein Pre-Reg-FAIL - sie qualifiziert es nur:
  - "RED weil definitiv nicht da"     vs
  - "RED weil Studie zu klein, nicht entscheidbar"
""")
