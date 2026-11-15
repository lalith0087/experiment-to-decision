# I built an experimentation toolkit, and the real data humbled it

I wanted a project that showed statistical rigor, not just model-fitting. So I built
[experiment-to-decision](../README.md): power analysis, sample-ratio checks, CUPED variance reduction,
always-valid sequential testing (mSPRT), uplift models, and a profit-based targeting layer. Then I ran it on
real data. Two of the things I expected to be impressive turned out to be null results, and those are the
parts I'm proudest of reporting honestly.

## 1. Peeking is the easiest way to fool yourself

If you check a normal z-test after every customer and stop when p < 0.05, you will "find" effects that do
not exist. I simulated 300 experiments with **no true effect**, checking after every observation up to
3,000 users:

| Test | False-positive rate |
|---|---|
| Naive z-test, peeking | **39%** |
| mSPRT (always-valid) | **2.7%** |

The nominal rate is 5%. mSPRT stays under it however often you look. The [live demo](https://experiment-to-decision.streamlit.app)
lets you change how often you peek and watch the naive rate climb.

## 2. On synthetic data everything worked. That proved little.

On simulated data with a known treatment effect, CUPED cut variance by about 68% and the uplift models
recovered the true effect with 0.95 to 0.99 correlation. These checks are useful, because they confirm the
code is correct (the test suite also checks that CUPED confidence intervals cover the truth 90 to 99% of the
time). But I built that data to be easy. A 68% variance reduction on a covariate I engineered to be predictive
says nothing about real life.

## 3. Hillstrom: a real, randomized email experiment

I ran the toolkit on the Hillstrom MineThatData experiment: 64,000 customers randomly assigned to a men's
email, a women's email, or nothing.

- **The experiment was healthy** (SRM p = 0.90). Any email lifted site visits by **6.1 percentage points**
  over a 10.6% baseline. The men's email gave +7.7 and the women's +4.5.
- **CUPED removed 0.4% of variance, not 68%.** Past purchase history barely predicts whether someone visits.
  CUPED is only as good as your pre-experiment covariate.
- **The uplift models beat random targeting**, but the T-, X- and doubly-robust learners were statistically
  indistinguishable (fold-to-fold variation was as large as their differences), and the decile table was noisy
  rather than cleanly monotone. There is weak but real heterogeneity in these 8 features.

I also tried to rescue CUPED with an ML-predicted covariate (CUPAC, boosted trees on all pre-experiment
features). Best case was 2.7% variance reduction on visits and under 0.2% on spend. The covariate's correlation
with the outcome was 0.15, and variance reduction cannot exceed that squared. So the disappointing CUPED number
was a limit of the data, not of the method.

## 4. The decision question: should we target, or email everyone?

This is where a model has to earn its keep. I built a policy layer that picks the share of customers to treat on
one half of the data, measures the profit on the other half, and bootstraps the whole procedure. Value was
spend, and I swept the cost per email from $0.10 to $0.70 (average spend uplift is about $0.60).

**Targeting did not provably beat emailing everyone at any cost level.** Every confidence interval for
"targeted minus email-all" included zero. At the highest cost, the point estimates favored targeting, but
64,000 customers with heavy-tailed spend can't confirm it. On this dataset the decision was driven by email
cost versus average uplift, not by the models.

### Was the heterogeneity even real?

A calibration test settles this. For visits, yes: the top-minus-bottom quintile gap is about +2.4 points
(95% CI 0.6 to 4.1), so there is genuine heterogeneity. But the models overstate it. Calibration slopes were
0.27 (T-learner), 0.58 (X) and 0.75 (doubly-robust); a slope of 1 means calibrated. For spend there was no
evidence of heterogeneity at all (every interval included zero), which is why the profit policy could not beat
treat-everyone. The negative result had a cause, and the test finds it.

## What I'd take from it

- Validate against known truth first, then expect real data to be much less kind.
- A method that reports calibrated uncertainty can tell you "we can't tell," which is often the right answer.
- A null result, reported with its interval, is more useful than an optimistic number.

## Limits

- Hillstrom is one dataset with one email campaign. Spend is heavy-tailed, so profit estimates are noisy.
- mSPRT uses a plug-in variance estimate and needs about 30 users per arm before it starts.
- The learners assume a randomized experiment (propensity equals the observed treated share).
- The learners are indistinguishable on Qini area; I cannot say which is best.

Code, tests and the demo: [github.com/lalith0087/experiment-to-decision](https://github.com/lalith0087/experiment-to-decision).
