# Summary

The paper introduces **STEMGYM**, an open-source, Gymnasium-compatible benchmark suite for dose-efficient autonomous scanning transmission electron microscopy (STEM). It contributes 15 physics-based HAADF-STEM simulated “worlds” spanning five materials and four task types, defines a unified dose-efficiency evaluation metric (DEC-AUC), and provides a suite of baseline agents and ablations to dissect the role of perception versus navigation/planning. Across >1,700 episodes, the authors find that perception quality dominates performance, with a trained analyst delivering a 5.8× improvement in DEC-AUC over naïve analysis, while more sophisticated navigation/planning yields negligible additional benefit, highlighting perception as the primary bottleneck.

# Strengths

## Technical novelty and innovation

- STEMGYM fills a clear gap by providing a standardized, open Gymnasium-compatible environment for autonomous STEM, integrating physics-based simulations (PRISM via abTEM) with annotated ground truth.  
- The Dose-Efficiency Curve (DEC-AUC) is a principled metric capturing the full trade-off between information gain and cumulative dose, rewarding both early convergence and final accuracy.  
- The causal decomposition design—equipping simple navigators with a shared analyst versus varying analysts on the same navigator—provides insightful attribution of performance to specific agent components.  
- Thoughtful task design spans multiple real-world goals (defect census, phase mapping, targeted characterization, particle census), improving generality beyond single-task benchmarks.

## Experimental rigor and validation

- Extensive evaluation across five materials, three difficulty levels, and four tasks with multiple agents and ablations (>1,700 episodes).  
- Difficulty scaling and dose-budget sensitivity analyses are informative, as is the cross-material transfer matrix that distinguishes within-family vs. cross-family generalization.  
- Ablations (e.g., uncertainty ensemble, planner variants) are carefully designed to isolate component contributions.

## Clarity of presentation

- The paper reads clearly and is well-structured, with a consistent narrative explaining the benchmark design, metric rationale, and key results.  
- Limitations (simulation–reality gap, the analyst’s specialization failure on particle census) are candidly discussed.

## Significance of contributions

- Provides a much-needed common ground for reproducible comparisons in autonomous STEM, likely accelerating progress and enabling fairer evaluation.  
- Identifies perception as the critical frontier, offering a concrete and measurable target for community improvement and method development.  
- Releases code, data, and configurations to promote adoption and transparency.

# Weaknesses

## Technical limitations or concerns

- The simulation–reality gap is significant: no inelastic scattering, stage drift, contamination, or non-Poisson noise are modeled; only an “experimentally motivated replay” suggests ranking preservation, which is limited in scope.  
- Navigation is modeled as dose-free and latency-free; real systems incur time and stability costs (hysteresis, drift, settling), which could alter the value of navigation strategies.  
- Fixed-dose acquisitions (no variable dwell or adaptive dose allocation within tiles) may underplay important dose-planning dimensions in real STEM.

## Experimental gaps or methodological issues

- Absence of reinforcement learning baselines is notable given the Gymnasium framing and claims about navigation/planning value; current agents are hand-crafted.  
- Limited details on the trained analyst: architecture, training data partitioning, data augmentation, potential leakage across worlds/difficulties, and hyperparameters are not fully specified; this weakens reproducibility and interpretation of the “perception bottleneck” claim.  
- Statistical testing is light: while standard deviations are provided, no formal significance testing is presented despite high variance for some settings; the number of seeds (2–5) may be small for some conclusions.  
- DEC-AUC’s sensitivity to scoring cadence (every 5 acquisitions) is not analyzed; different scoring intervals can affect area estimates and comparisons.

## Clarity or presentation issues

- Some tables contain formatting artifacts (e.g., intended dose notations in Table 3), likely due to PDF extraction; while minor, a clean version would help.  
- The particle census section shows dramatic failure for equipped agents; while discussed, more clarity on why the scoring goes to zero (e.g., incompatibility between analyst output schema and task API) and how to remediate would be useful.

## Missing related work or comparisons

- Comparisons to dose-efficient sparse scanning/compressed sensing in STEM (e.g., dynamic sparse sampling, inpainting reconstructions) are missing; including such baselines would contextualize the value of navigation vs. reconstruction.  
- Limited discussion of related autonomous microscopy work in scanning probe and synchrotron domains leaves some cross-domain opportunities underexplored (e.g., action costs, active learning formulations).  
- The planned VLM comparison is incomplete; claims about zero-shot VLM potential remain speculative without results.

# Detailed Comments

## Technical soundness evaluation

The benchmark construction is technically solid, using scientifically reasonable simulation parameters (energies, detector angles, Debye–Waller factors) and structured defect distributions to reward adaptive behavior. The DEC-AUC metric is well-motivated for dose-limited settings.

The action space and observation design are plausible abstractions, but the assumption of free and perfect navigation (no time/instability costs) simplifies away real-world constraints that often shape acquisition strategies.

The conclusion that perception dominates appears sound within the present setup; however, the absence of learned navigation policies and the simplified navigation cost model may limit the generality of this conclusion.

## Experimental evaluation assessment

Broad coverage across materials, tasks, and difficulties strengthens the conclusions. The ablations are insightful, particularly the role of the uncertainty ensemble in signaling phase transitions.

The dramatic inversion on particle census (naïve raster scoring ~0.91 vs. 0.00 for equipped) is important: it underscores the danger of single-analyst designs across heterogeneous tasks and suggests the need for multi-head or task-adaptive perception modules.

Cross-material transfer experiments are valuable. Strong within-family generalization for perovskites is encouraging and practically actionable, while poor transfer to wurtzite/diamond-cubic systems highlights the need for more diverse training.

Statistical rigor could be improved: provide confidence intervals or formal tests for the main claims; increase seed counts where variance is high; examine sensitivity of DEC-AUC to scoring cadence and dose budgets beyond the three explored.

## Comparison with related work

The benchmark addresses the lack of standardized environments and datasets highlighted in broader microscopy and automation literature and is conceptually similar to benchmark efforts in cryo-EM and other scientific domains.

However, comparisons to dose-efficient imaging approaches (e.g., compressed sensing, sparse sampling + reconstruction) are missing and would strengthen the framing, as these represent alternative routes to dose reduction with distinct trade-offs.

Closer alignment with autonomous experimentation frameworks in synchrotron and scanning probe microscopy communities could inspire more realistic cost models and active learning formulations, particularly conditioning acquisition policies on analysis outputs.

## Broader impact and significance

A standardized, open benchmark for autonomous STEM is timely and impactful. It should promote reproducibility, fair comparison, and faster methodological progress.

The identification of perception as the primary bottleneck provides a focused research agenda. Practical guidance, such as within-family transfer for perovskites, is actionable.

For real-world deployment, bridging the sim-to-real gap and incorporating realistic navigation costs and noise processes will be essential. Expanding to real or semi-synthetic datasets would improve external validity.

Releasing code, worlds, and checkpoints under permissive licenses is a positive step.

# Questions for Authors

1. How is the analyst trained and validated? Please detail training/validation/test splits across worlds and difficulties to rule out leakage; include architectures, augmentations, and hyperparameters for reproducibility.  
2. How sensitive is DEC-AUC to the scoring cadence (every 5 acquisitions)? Have you evaluated alternative cadences or continuous-time approximations to ensure ranking stability?  
3. Can you quantify the effect of navigation latency or instability by introducing a time budget or movement penalty, and assess whether the “navigation adds little” conclusion holds under more realistic costs?  
4. For particle census, what precisely causes the equipped agents to score zero (schema mismatch vs. misclassification)? Could a simple analyst switch or multi-head model recover performance, and do you plan to include such a baseline?  
5. Do you plan to include learned RL navigation baselines and/or active learning approaches that condition acquisition on analyst outputs (e.g., GP over defect likelihood)?  
6. Could you provide a small real HAADF-STEM evaluation set (with annotations) to assess sim-to-real transfer and ranking preservation more convincingly?  
7. How do results change with variable per-tile dose allocations (adaptive dwell), an important practical degree of freedom in dose management?

# Overall Assessment

This paper presents a well-executed and necessary benchmark for autonomous, dose-efficient STEM. The combination of simulated, annotated worlds; task-diverse scoring; the DEC-AUC metric; and a structured evaluation suite makes STEMGYM a valuable resource. The central empirical insight—that perception dominates navigation/planning under the current assumptions—is supported within the benchmark and provides direction for future work.

Limitations remain, particularly the simulation–reality gap, simplified navigation cost model, absence of RL baselines, and the analyst’s failure on particle tasks. Nonetheless, the authors acknowledge many of these constraints and position STEMGYM as a development benchmark. With clearer analyst training details, stronger statistical analysis, and more realistic cost modeling or learned navigation baselines, the work would offer substantial value to the NeurIPS Datasets and Benchmarks community.

I recommend acceptance, contingent on the above improvements.