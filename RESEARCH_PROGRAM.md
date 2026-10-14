# Research program for the Nature Human Behaviour paper

Written 14 October 2026. This is the map of the paper. Each analysis has a question, a reason, a method a first time reader can follow, and the one sentence message it is meant to deliver. `ANALYSIS_PLAN.md` freezes the confirmatory parts before their data are seen, and `STORY.md` logs every result.

## The story we are testing

Choice overload is famous and unpredictable. Meta-analyses find an average effect near zero with large differences between studies (Scheibehenne, Greifeneder and Todd 2010) and point to moderators such as set complexity and task difficulty (Chernev, Böckenholt and Goodman 2015). No model says in advance which displays will overload people.

Our claim is that people search, and that search explains overload. They inspect options in order, stop when one is good enough, and choose more selectively in larger sets rather than more randomly (H1, confirmed on open snack data). If that is how people choose, a model that searches the same way should predict three things the standard models cannot, namely what people choose, how long they take, and when they will be overloaded.

## The three primary analyses

### 1. Premonition of choice overload

Question. Before anyone chooses, can we tell which displays will overload people?

Why. A model that captures how people choose should also know when choosing gets hard. This is the practical payoff, since designers, retailers and defenders could then check a display before showing it.

Measuring overload, per person and trial. The study asked six questions after every trial, namely satisfaction, certainty of having chosen best, confusion, frustration, complexity and certainty again. These are the classic overload measures. We combine them into one overload index (signs aligned, standardized within each person so the index describes the trial, not the person) and keep two objective signs beside it, regret (100 minus the rubric score of the chosen option) and decision time.

Predicting it. Every predictor is computed from the display alone, or from a model fitted to other trials, so nothing about the predicted trial's choices or ratings is used.
- Display features. Number of options, attributes shown, number of options close to the best, number of options no other option beats on every attribute.
- Model forecasts. Each model's expected spread of choices, expected regret and expected effort for the held out trial.

Test. Which forecast best predicts the overload index, regret and time of unseen trials, beyond the number of options.

Message we expect. Overload is predictable from the structure of a display through a model of how people search.

### 2. How long each model takes to decide

Question. How long would each model take to make the choice, and which one matches how long people take?

Why. A model that chooses like people should also take as long as people. Time is evidence about the process that choices alone cannot give.

Method. Each model is written as the elementary steps it needs (Payne, Bettman and Johnson 1993). Reading an attribute, comparing two values, adding, eliminating. Counting those steps for every display gives the model's effort. Predicted time is a fixed start up time plus a time per step, the same two timing constants for every model and fitted on other trials, so models differ only in the work their algorithm does.
- CL reads and weighs every attribute of every option.
- DbS and MLBA compare every option with every rival on every attribute.
- Elimination by aspects stops as soon as one option survives.
- The new model stops when it accepts an option.

Test. Held out error in decision time per trial, and whether each model reproduces how time grows with the number of options and attributes.

Message we expect. People's time grows far more slowly with set size than integration models require, and a search that stops matches it.

Outcome, 14 October 2026. Not supported. Fitted to people's times, no model's work, ours included, predicted decision time better than a constant on the development data, or better than the number of options on the snack data. Dividing people's time by a model's steps is not a fit and is not used.

### 3. A model that decides like people

Question. Can one mechanism, with one free parameter like the other models, beat them on choices and on time?

Proposed mechanism, search and accept. The decision maker glances at the display, then inspects options in reading order as each display showed them (from the storyboard). Each inspected option is accepted when its noisily perceived value reaches the best value on display, which happens with probability 1 / (1 + exp(-lambda (v_i - v_max))), using the same rubric values as CL. The best options are accepted half the time and poor ones rarely. If an option is rejected, search moves to the next one, and passes repeat until an option is accepted, so every option keeps some chance.
- One free parameter, lambda, as in every other model.
- Choice probabilities, expected search length and expected time follow in closed form.
- It should produce narrowing (search stops early in rich displays), sharpening (mediocre options are rarely accepted when good ones abound), primacy among good options (the concentration and listing position effects of the development data) and overload (long searches when few options come close to the best). Each of these is checked, not assumed.

Test. Held out completeness of choices, held out error in time, and premonition of overload, against QR, CL, DbS, MLBA and elimination by aspects. On the snack data, search length is also checked against how many items people actually looked at.

Message we expect. People are not noisy integrators. They search and accept, and a model that does the same predicts their choices, their time and their overload.

## Completeness across datasets

Completeness stays the yardstick for choices. It needs the same display shown to many people. Where every display is new, held out likelihood relative to chance replaces it.

| Dataset | Options | What it adds | Status |
|---|---|---|---|
| Development, file selection and laptop purchase | 8 to 32, brief or full descriptions | overload ratings, regret, time, completeness | in hand, discovery data |
| Snack choice (Thomas, Molter and Krajbich 2021) | 9 to 36 | time and eye movements, search length | in hand, choices already used, time not yet looked at |
| Search and satisficing (Caplin, Dean and Martin 2011) | 10 to 40, simple or complex values | the order people search and when they stop, same value sets reused | open on openICPSR, needs a free download |
| Eye tracking under time pressure (Reutskaja, Nagel, Camerer and Rangel 2011) | 4 to 16 | search under time pressure | open on openICPSR, to check |
| Gaze bias datasets (Thomas, Molter, Krajbich, Heekeren and Mohr 2019) | 2 and 3 | time at the smallest set sizes | open on GitHub |

## Order of work

1. Close the laptop correction. Verify the rerun laptop results twice and update R0 and H1.
2. Overload measures on the development data. Build the index, check that the six questions measure one thing, and show which trials overload people.
3. Decision time. Count each model's steps, fit the two timing constants, compare with people.
4. The search and accept model. Fit, compare on choices, time and overload.
5. Premonition. Forecast overload from each model, held out.
6. Freeze the confirmatory predictions by amendment, then test them once on the snack data's time and on the open datasets.

The development data are discovery data, so steps 2 to 5 are exploratory there. The claims rest on step 6.

## How every analysis is reported

Each notebook section and each module opens with its question and message. Findings cells state the result in plain words with intervals and end with what it means for the story. Every number is verified by a second run and by an independent implementation before it is shown. No participant counts appear in working outputs.
