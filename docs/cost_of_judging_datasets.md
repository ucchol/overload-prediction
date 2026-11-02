# Open datasets for a set size by cost of judging test

Search date 2026-11-02. Everything below was checked on a primary page (journal page, author working paper, repository record, or publisher supplement). Where a page could not be opened from this environment, the entry says so. No data were downloaded. The OSF API and several OSF project pages were blocked by the network proxy, so OSF links quoted from papers are reported as stated by the paper, not as inspected.

## Bottom line

No open dataset I could verify is a strong fit. None crosses three or more set sizes with a clean cost of judging manipulation, within participants, with many trials per person and a public file whose contents I could confirm. The closest open one is Aguiar, Boccardi, Kashaev and Kim (2023). It crosses menu size with an arithmetic cost manipulation in the Caplin, Dean and Martin style, and the journal page lists a code and data zip. Its weak points are few choices per person, a risk based value (expected value, not a sure amount), and menu size that varies only partly within a person. The two best designs I found (Duffy, Gussman and Smith 2021 and Jessup, Ritchie and Homer 2020) do not have a public data file I could confirm. A short preregistered online study is the cleanest route, and a sketch is at the end.

## Top three

1. Aguiar, Boccardi, Kashaev and Kim (2023). Partial fit, but open. This is the only verified public file that crosses set size with a computed value manipulation.
2. Duffy, Gussman and Smith (2021). Strong design and partial fit overall, but the data are unverified. The paper links an OSF project for screenshots only. Ask the authors.
3. Jessup, Ritchie and Homer (2020). Strong design match on time pressure, but time pressure is between participants and the data are not public. Ask the authors.

## Candidates

### 1. Aguiar, Boccardi, Kashaev and Kim (2023), Random utility and limited consideration

- Citation. Quantitative Economics 14(1), 71 to 116. DOI 10.3982/QE1861.
- Repository. The journal supplement page lists `QE1861_code_and_data.zip` and an online appendix PDF at https://www.econometricsociety.org/publications/quantitative-economics/2023/01/01/Random-utility-and-limited-consideration . No Zenodo or Dataverse DOI is listed. I did not open the zip, so the file contents are unconfirmed.
- License. None is stated on the supplement page beyond the Econometric Society copyright and terms of use link. Check before reuse.
- Design. Online on MTurk in 2018. There are five lotteries plus a sure outside option worth 12 tokens, which is always shown first and preselected. Prizes in each lottery had to be computed from a chain of two digit additions or subtractions, with chain length k = 1, 3 or 5 (low, medium and high cost). The outside option needed no arithmetic. Menus cover all nonempty subsets of the five lotteries, so set size runs from 1 to 5 lotteries plus the outside option. Each person saw at most two disjoint menus (either the full set or a split of the five lotteries into two menus), and each menu was shown under all three cost levels. So cost is within participants, and set size is partly within (two sizes at most per person) and mostly between.
- Objective values. Yes in expected value terms. Lottery expected values run from 21.6 to 25 tokens, and the outside option has the lowest expected value. The top expected value lottery (l1) also has the highest variance, so "best" depends on accepting expected value as the yardstick.
- Choices per trial. Yes. Each choice is a separate observation.
- Trials. About 12,300 choices in total, at most about six per person.
- Paper findings relevant here. Choice of the outside option rises roughly linearly with arithmetic cost, and the paper reports a positive net effect of menu size on the outside option at medium and high cost (its Table 12), though its figures and text are not fully consistent on this.
- Fit. Partial. It has the right manipulation (computed values crossed with menu size), but the noisy risk based value, few trials per person, and mostly between person set size make it a weak confirmatory test.

### 2. Duffy, Gussman and Smith (2021), Visual judgments of length in the economics laboratory

- Citation. Journal of Behavioral and Experimental Economics 93, 101708. DOI 10.1016/j.socec.2021.101708. Working paper versions are MPRA 93126 (2019) and MPRA 106355 (2021).
- Repository. The working papers link https://osf.io/srpzh/ for the full set of screenshots and do not say raw data are posted there. I could not open the OSF page from here.
- License. Not verified.
- Design. Participants chose the longest of 2, 3, 4, 5 or 6 lines (equal probability) and were paid by the length of the chosen line. Only one line could be viewed at a time. The cost manipulation was memory load, with a 6 digit number (high load) or a 1 digit number (low load) held during each choice. Load was mixed randomly across trials. There was a fixed 15 second selection window. Set size and load were both within participants.
- Objective values. Yes. Line length in pixels is the paid value.
- Choices per trial. Yes.
- Trials. 100 choices per person, about 9,200 in total.
- Fit. Strong on design and partial overall. It has five set sizes crossed with a cost manipulation inside each person, at-a-glance values, and an objectively best option. The concerns are that memory load taxes holding values rather than judging each one, and that the data file is not confirmed public.
- Related. Duffy and Smith (2025), An economist and a psychologist form a line, Theory and Decision 99(3), 701 to 734, DOI 10.1007/s11238-025-10040-4. It also uses 2 to 6 lines and crosses them with easy, medium and difficult length gaps. Its supplement reports accuracy falling from about 78 percent with 2 lines to about 63 percent with 6. Difficulty there is the value gap, not a cost of judging, so it fits less well. The supplement links https://osf.io/f7gu4/, but that project holds a different dataset (triples). Data for the main study were not verified. A related binary line study (Duffy and Smith 2025, Journal of Economic Psychology, DOI 10.1016/j.joep.2024.102787) states data at https://osf.io/us9n6/ but has only two options, so it does not qualify.

### 3. Jessup, Ritchie and Homer (2020), Hurry up and decide

- Citation. Decision (APA). DOI 10.1037/dec0000115. An author copy is at https://digitalcommons.acu.edu/psychology/4 under a CC BY-NC-ND 4.0 license (this covers the article, not data).
- Repository. None. The paper points only to APA online supplements (doi 10.1037/dec0000115.supp) for applicant quality details and instructions. There is no OSF link or data statement.
- Design. A virtual hiring task with 3 versus 15 applicants (within participants), crossed with time pressure of 5 seconds versus 120 seconds per decision (between participants). Quality was either exponentially distributed (one applicant tends to dominate) or uniformly distributed (within). A no-hire option was available.
- Objective values. Yes. Each applicant's quality was computed from a linear model of four attributes. Feedback gave a noisy estimate of quality.
- Choices per trial. Yes, hire one or no hire, with response times.
- Trials. 80 decisions per person (20 per set size by distribution cell).
- Fit. Partial. It is the closest published design to the hypothesis (time pressure by set size with known quality), but it has only two set sizes, time pressure is between participants, and the data are not public.

### Other candidates checked

- Besedeš, Deck, Sarangi and Shor (2012), Age effects and heuristics in decision making, Review of Economics and Statistics 94(2), 580 to 595. Options covered states of a known card draw, and the best option is the one covering the most probability. The design crossed 4 versus 13 options with 6 versus 10 states (more states means more to add up per option), within participants, with 8 decisions per person. No data statement was found on the working paper or the author's page, so data are unverified. Partial on design, weak on availability.
- Besedeš, Deck, Sarangi and Shor (2012), Decision making strategies and performance among seniors, Journal of Economic Behavior and Organization. It is the same task with a 3 by 3 within person design of 4, 8 or 12 options by 4, 8 or 12 states, and 18 tasks per person. It is paper and pencil, with no time limit. The working paper says only that the instrument is available on request. This is a good design with no public data, so it is weak on availability.
- Besedeš, Deck, Sarangi and Shor (2015), Reducing choice overload without reducing choices, Review of Economics and Statistics 97(4), 793 to 802. Harvard Dataverse record DOI 10.7910/DVN/27616 (the record exists, but I could not open the Dataverse page to check files or license). Options were 16 at once versus rounds of 4, with no cost manipulation and four tasks per person. Weak.
- Dean, Ravindran and Stoye, A better test of choice overload (arXiv 2212.03931, working paper dated 2026). This is the closest Caplin, Dean and Martin follow-up. Options were written as sums of four numbers from 0 to 10, with a preselected default worth 7. Menus held 2 or 3 options or the full 13. There was no cost manipulation, each person made 10 choices, and no data link appears in the paper. Weak for this test, though useful as a set size benchmark with computed values.
- Gonçalves, Nunnari and Zarate-Pina, Revealing complexity (extended abstract, May 2024). It describes a Caplin, Dean and Martin style arithmetic task that varies both the number of operations and the number of alternatives. Only pilot results are reported, with no data or design details. It is worth watching or emailing the authors, since this is the exact cross.
- Gabaix, Laibson, Moloche and Weinberg (2006), AER 96(4), 1043 to 1068. A Mouselab task with 8 alternatives and 10 attributes, crossed with exogenous versus endogenous time. Set size is fixed and the data were not verified as public (Sanjurjo 2017 obtained them from the authors). Weak.
- Bhargava, Loewenstein and Sydnor (2017), Choose to lose, QJE 132(3), 1319 to 1372. Experiment 1 crosses 4 versus 12 health plans with monthly versus annual premium display and sequential versus table interface, with dominance known by design. It is between participants with one choice per person, and the data were not verified. Weak.
- Carvalho and Silverman, Complexity and sophistication (NBER w26036). Complexity is the number of redundant assets (2 versus 5) and is between participants. There is no separate cost manipulation and no data statement. Weak.
- Fan et al. (2025), Overt visual attention in the formation of preference between complex lottery options, Computational Brain and Behavior, DOI 10.1007/s42113-025-00239-7. Data at Johns Hopkins Research Data Repository DOI 10.7281/T1/7IXSOP (license not stated in the paper). It compares a 2 option by 2 attribute task with a 4 option by 4 attribute task, so set size and attribute load are confounded, with no time limit. Weak.
- Yeon and Rahnev (2020), The suboptimality of perceptual decision making with multiple alternatives, Nature Communications 11, DOI 10.1038/s41467-020-17661-z. Data and code at https://osf.io/d2b9v/ per the paper. It covers 2 versus 3, 4 or 6 alternatives with brief stimuli, but difficulty is not crossed with set size as a cost manipulation and values are categorical, not graded. Weak.
- Gluth, Kern, Kortmann and Vitali (2020), Nature Human Behaviour, DOI 10.1038/s41562-020-0822-0. Data at https://osf.io/qrv2e/ per the article. Every trial has three snacks with subjective values and no set size or cost variation. Weak.
- Seo, Lee, Lee and Kim (2026, bioRxiv DOI 10.64898/2026.02.19.706731) and Huang and Xu (2025, NeuroImage, DOI 10.1016/j.neuroimage.2025.121542). Both cross set size (6 to 24 posters, or 4 versus 16 snacks) with a demand or preference factor, but values are subjective ratings and data are available only on request. Weak.
- Schram and Sonnemans (2011), How individuals choose health insurance, European Economic Review 55(6), 799 to 819. Number of policies was varied, but there is no separate cost manipulation and no data statement on the abstract page. Weak.

### Follow-ups of the datasets already in use

None of the follow-ups I found qualifies. Dean, Ravindran and Stoye reuse the Caplin, Dean and Martin computed value idea with set size but no cost factor, and their data are not posted. Gonçalves, Nunnari and Zarate-Pina run the exact cross but only as a pilot. Besedeš et al. (2015) is a choice architecture follow-up in the Reutskaja style but has no cost factor. The Thomas, Molter and Krajbich and Gluth lines of work use subjective snack values and do not vary the cost of judging.

## Minimal preregistered online study

If none of the above holds up, a single session online study can test the hypothesis directly.

- Task. Choose the highest valued card from a grid. Each card's value is paid as a bonus on one randomly drawn trial, so the best option is known exactly.
- Set size. Use 2, 4, 8 and 16 cards in a fixed grid with unused slots shown as blanks, so screen layout stays the same across set sizes.
- Cost of judging, within participants. In the low cost condition each card shows its value as one number (for example 47). In the high cost condition each card shows a sum of three two digit numbers padded to the same width (for example 12 + 9 + 26), as in Caplin, Dean and Martin. An optional middle level uses one addition. Keep fonts, card size and colors identical across conditions.
- Value structure. Draw values so that the gap between the best and second best card and the spread of the remaining cards are the same at every set size. Use a few gap levels so the analysis can separate difficulty from overload.
- Time. Allow free response with a generous cap (around 60 seconds) and record response times, so the low cost condition can show compensation (longer time, flat accuracy) and the high cost condition can show lost accuracy. A second preregistered variant can swap arithmetic for a deadline as the cost factor.
- Trials. 4 set sizes by 2 cost levels, with about 20 trials per cell, gives about 160 trials. Block cost in counterbalanced order and randomize set size within blocks. Add a short arithmetic practice and attention checks.
- Primary test. A mixed effects logistic model of choosing the best card on log set size, cost and their interaction, with random slopes by participant. The prediction is a negative interaction, where accuracy falls with set size under high cost and stays flat under low cost while response time rises. Secondary outcomes are value lost relative to the best card and response time.
- Sample size. Set it by simulation from a small pilot, targeting the interaction term.
