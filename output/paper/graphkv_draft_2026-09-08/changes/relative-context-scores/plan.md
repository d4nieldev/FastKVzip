# Figure 2 relative scores

User request (verbatim):

> in the paper, figure 2 shows the score as a function of the context across mixed benchmarks. Because the benchmarks are mixed and each benchmark has a different difficulty I think it is better to show the relative score instead of just the raw score

The user confirmed the normalization preference: “Yes—relative to each method’s full-cache score”.

Implementation scope (agent interpretation of the request, not a separately approved plan): normalize within each configuration and length bucket, preserve context-count weighting and the matched examples, update Figure 2 and its explanation, and rebuild the paper. Check normalization against the saved scores and visually inspect the affected PDF pages. Further choices and validation are recorded in `decisions.md`.
