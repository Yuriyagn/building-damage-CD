# BRIGHT supplement near-overlap review

The focused audit compared all 262 BRIGHT supplement candidates against the 290 event-group-v1 validation samples and 290 test samples.

- High-risk candidates: 0
- Medium-risk candidates: 3
- Train-validation candidates retained at score >= 0.60: 0
- Confirmed overlaps after visual review: 0
- Supplement samples removed: 0

All three medium-risk pairs were false positives caused primarily by similar diagonal no-data boundaries. Two pairs had zero RANSAC inliers; the third had only two. The visible geography, road geometry, buildings, and land cover are different in every pair.

Decision: **pass after human review**. Keep all 262 supplement samples.

The 5.6 MB review contact sheet is intentionally excluded from the source repository;
the compact decision record is preserved in `review_summary.json`. Rebuild the visual
sheet from the overlap-audit workflow when image-level inspection is required.
