**Visual reading of the galleries** for the direct change U-Net. These are my interpretation of the 20 thumbnails above: the pixel shares in the table are computed, these notes are not.

*False positives*, where the model says loss and Hansen says no loss 2019–2022:
- **Boundary slivers** along Hansen loss patches and forest edges (#5–#10). Hansen's 30 m pixels and the 10 m imagery disagree by 1–3 pixels. These are much thinner than with differencing.
- **Degraded or burned forest around a clearing** (#1), and a patch that **PRODES maps as 2019–2022 deforestation** (#4). In both the canopy looks thinned in 2022. The model may be right and the Hansen label incomplete.
- **Real false alarms inside intact forest** (#2, #3). Both years look like closed forest.

*False negatives*, where Hansen says loss and the model misses it:
- **Edge strips of Hansen loss patches** (#2, #4, #5, #7–#10), with little visible change at 10 m. This matches the computed category shares (mostly "edge").
- **Partly detected clearings** (#3) and **degraded rather than cleared forest** (#1).
- A narrow tree line in pasture (#6).
