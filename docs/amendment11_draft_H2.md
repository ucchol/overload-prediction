# Draft, Amendment 11. H2, the return rule, tested on gaze sequences

Status. Draft for the study's author to review. Nothing below has been run on recorded gaze or choices.

Claim. People search by letting attention return to attractive options, and the option attention returns to first is the one chosen. Choices alone cannot separate this from CL (Amendment 5), so the test uses the order of looks, which CL does not describe.

What has been seen. Snack data, set size means of returning gazes and distinct items looked at (Amendment 5), and the first rows of one aggregate file. The per person gaze files (`data/subject_files/*_desc-gazes.csv`, columns subject, setsize, trial, item, dur, onset, stimulus, gaze_num, is_returning, returning_gaze_count, is_last, is_first, item_value, choice, is_last_to_choice) have been listed and one header read, nothing else. Reutskaja data, Amendment 8 computed the share of misses whose best item was never fixated, by set size, so a relation between fixations and choices has been seen there. The published papers report that the last item looked at was chosen on 44.5 percent of snack trials (Thomas, Molter and Krajbich, 2021) and that early search is close to random with respect to value under time pressure (Reutskaja and colleagues, 2011). Before freezing, `scripts/simulate_return_rule.py` showed on the real snack displays that the test separates a partial return rule (return cue +0.86 bits, dwell +0.30), choice by dwell (return +0.03, dwell +0.12) and gaze unrelated to choice (both 0.00).

1. Data. The snack per person gaze files at the commit of Amendment 3, and Reutskaja's fixations (`Roinum` in `fixOrder`), consecutive fixations on one item merged into one gaze, as Thomas and colleagues define a gaze.
2. Cues per trial, from the gaze sequence. Return, 1 for the first item whose gaze is a return to an item already gazed at, 0 otherwise (all 0 when no item is returned to). Dwell, each item's share of the trial's total gaze time (the cue of gaze weighted accumulation models such as GLAM). Last, 1 for the last item gazed at.
3. Models. CL on the ratings, and CL plus one cue, softmax(lambda v + beta cue), lambda >= 0, beta free, by maximum likelihood. Each set size is predicted from fits to the others (leave one set size out). The score is bits per choice.
4. Primary test, in each dataset. Gain of the return cue over CL, and the return cue's gain minus the dwell cue's. H2 is supported in a dataset when both are clearly positive, and supported overall when it holds in both datasets.
5. Secondary and descriptive. The last cue's gain (the end of a trial is close to the response, so it is reported, not tested). The share of trials in which the first returned item was chosen, against CL's probability for that item, by set size. The share of trials with no return. Each cue's fitted beta.
6. Resampling. Participant bootstrap, B = 1000, numpy.random.default_rng(110000 + b), snack first then Reutskaja, everything refitted in every fold of every replicate. Intervals and the clear rule follow Section 4.
7. Reporting. The two primary tests are confirmatory. The 3.6 fold gap between how many items people and the model look at is reported beside them. A pass means the return carries information about the choice beyond value and beyond time spent looking. It does not mean people stop at the first return.

Open points for the study's author.
- Requiring both datasets is strict. The alternative is to call H2 supported in each dataset separately.
- Dwell time is the stronger competitor (it is what gaze models use). Number of gazes is the alternative.
