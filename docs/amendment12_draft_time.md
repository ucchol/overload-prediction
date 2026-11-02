# Draft, Amendment 12. Decision time from trial structure, across every dataset with times

Status. Approved 2 November 2026 and frozen as Amendment 12 in ANALYSIS_PLAN.md, which is the authoritative text.

Why a new test. Earlier time tests asked whether a model's internal work (comparisons, looks) explains time, and none beat a constant or counting options. Every fitted law put most of the time into a start up constant, which is mostly how fast each person is. This amendment asks a different and practical question. Given only the structure of a trial (how many options, how good the best is, how close the runner up is), can we predict which trials take longer for a given person, within a dataset, and in a dataset the law has never seen? It also asks once more, on every dataset, whether the search model's expected looks add anything beyond that structure.

What has been seen. Times have been used in every dataset below (Sections S2, G1, the development time laws, and the search time of Amendment 7). Known before this plan. Times grow with set size (snack, Caplin, Dean and Martin, elasticity about 0.24 to 0.29). The best item's value foresees slow decisions (-0.32 on snack, -0.20 on the gaze sets). Within person centering, the transferred law and the three tests below have not been computed on any dataset.

1. Datasets and times. Eight sets, every one with trial times and option values. Development files and development laptops (seconds per trial, rubric values 0 to 100), snack (rt), Reutskaja and colleagues (rt, all within the three second limit), Caplin, Dean and Martin (time of the final selection, the rounds of Amendment 7), Krajbich 2010, Krajbich 2011 and Folke 2016 (rt, values 0 to 10 as prepared by Thomas and colleagues). The lottery data hold no times and are not used. Inclusion follows each dataset's earlier amendment.
2. Outcome. Log time in seconds, centered within participant (minus that person's mean log time over their included trials). Centering removes how fast each person is, so the question is which trials take longer.
3. Trial structure. Values are rescaled to 0 to 1 by each dataset's value scale. Four features per trial, log2 n, the best value, the gap between the best and second best, and the mean of the other options. Each is centered within participant. log2 n is dropped where n does not vary within a dataset.
4. Laws. S1, log2 n alone. S2, the four features. S3, S2 plus the log of sample until return's expected looks. S4, S2 plus the entropy of CL's choice probabilities (a competing model signal). The sensitivities of S3 and S4 are fitted to the training participants' choices. Coefficients by least squares, no intercept (the outcome and features are centered).
5. Validation. Within a dataset, ten folds of participants as in Amendment 6. Across datasets, the S2 coefficients are fitted on the pooled centered data of the other seven sets, each set weighted equally, and applied unchanged to the held out set (the transferred law). The score is the held out share of variance explained, R2 = 1 - SSE / SS of the centered log time.
6. Primary tests, each across the eight sets with Holm's correction.
   - T1, structure beyond counting. R2 of S2 minus R2 of S1, supported in a set when clearly positive.
   - T2, transfer. R2 of the transferred law, supported in a set when clearly positive.
   - T3, the search model beyond structure. R2 of S3 minus R2 of S2, supported in a set when clearly positive.
7. Descriptive. The coefficients per set and of the transferred laws. S4 against S2. For sets with several set sizes, the growth of mean time per doubling of the set, as observed, as fitted with that set size held out, and as transferred. Each set's R2 of S2 itself.
8. Resampling. Participant bootstrap, B = 1000, numpy.random.default_rng(120000 + b), sets drawn in the order of item 1, everything refitted in every fold of every replicate. Intervals and the clear rule follow Section 4.
9. Reporting. All eight sets are reported for every test whatever the result. No set is chosen after the run.

Equivalence for snack growth (descriptive, Section B).

The snack growth of misses per doubling is +0.010 with a 95 percent interval of -0.003 to 0.026, already known. A two one sided test with bound delta asks whether the 90 percent interval lies inside -delta to +delta. Because the interval is known, the test cannot be confirmatory, so it is reported as descriptive with the bound and its reason stated.

Checks before freezing. The code is run on times simulated from S2 with known coefficients in every dataset's real trials (T1 and T2 should be supported) and from S1 alone (T1 should not be).
