# bayesian-model-comparison
This repository provides Python implementations of various Bayesian model comparison methods. 
These models facilitate the comparison and ranking of machine learning algorithms considering different data structures and assumptions. 
The implementation is primarily based on three repositories: [scmamp](https://github.com/b0rxa/scmamp/tree/master) ([Calvo and Santafé Rodrigo, 2016](https://journal.r-project.org/archive/2016/RJ-2016-017/RJ-2016-017.pdf)), 
an R implementation for statistically analyzing the results of algorithm comparisons across different problems, 
[baycomp](https://github.com/janezd/baycomp/tree/master) ([Benavoli et al., 2017](https://www.jmlr.org/papers/volume18/16-305/16-305.pdf)), a Python library for Bayesian comparison of classifiers,
and [cmpbayes](https://github.com/dpaetzel/cmpbayes), a small Python library that provides tools for performing Bayesian data analysis on the results of running algorithms.


The models implemented in this repository are listed below:

| **Model**                                                          | **# Algorithms** | **# Datasets / Tasks** | **Data Structure**                                                             | **Parametric Assumption**                                                                                              | **Clarified Description / Key Strengths**                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------ | ---------------- | ---------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Bayesian Beta-Binomial Test** (von Pilchau et al., 2023)         | 2                | ≥ 1                    | Independent runs (per-task posterior combined trivially across tasks)          | No normality needed – each algorithm’s success count is modelled with a conjugate Beta prior and a Binomial likelihood | Provides an exact posterior for the difference in success probabilities, automatically accounting for small‐sample uncertainty and class imbalance. Delivers intuitive quantities such as *$P(A > B)$* and the full posterior of the risk ratio; credible intervals tighten as evidence accumulates.                        |
| **Bayesian Correlated t-Test** (Corani & Benavoli, 2015)           | 2                | ≥ 1 (folds)            | **Correlated** within each dataset due to overlapping CV folds                 | Yes – paired differences assumed Student-t; correlation ρ estimated analytically                                       | Extends the paired *t*‐test by treating fold-level differences as a correlated multivariate normal. Outputs the posterior of the mean difference plus the probability that it lies in a user-defined ROPE. Calibrated for *k*-fold/ repeated CV and often paired with a Poisson-binomial layer for multi-dataset inference. |
| **Bayesian Hierarchical t-Test** (Corani et al., 2017)             | 2 +              | Multiple               | **Hierarchical** – per-dataset effects $\delta_1$ are correlated via shared hyper-priors | Yes – normal likelihood with shrinkage priors                                                                          | Pools information across datasets by modelling $\delta_1$ jointly. The global variance hyper-prior shrinks extreme per-dataset estimates, yielding narrower credible intervals and direct probabilities of practical equivalence (via ROPE). Handles missing folds and varying numbers of CV repetitions naturally.         |
| **Bayesian Independent t-Test (BEST)** (Kruschke, 2013)            | 2                | ≥ 1                    | Independent samples                                                            | Yes – Student-t likelihood with unknown ν (robust to outliers)                                                         | Produces full posteriors for group means, standard deviations and standardized effect size *d*. Allows acceptance of the null (difference within ROPE) and delivers power curves without relying on asymptotics. Recommended when data satisfy i.i.d. assumptions.                                                          |
| **Bayesian Non-Negative Bimodal Test** (cmpbayes)                  | 2                | ≥ 1                    | Independent                                                                    | No – two-component Gamma mixture for strictly non-negative data                                                        | Tailored to metrics such as runtimes where a minority of “fast-path” observations create a second mode. Each algorithm is modelled by a Gamma–Gamma mixture; posterior integration over mixture weights yields *$P(A < B)$* etc. Robust to heavy right tails and zero-inflation.                                            |
| **Bayesian Non-Negative Unimodal Test** (von Pilchau et al., 2023) | 2                | ≥ 1                    | Independent                                                                    | No – single Gamma (or Log-Normal) likelihood; optional right-censoring                                                 | Simpler sibling of the bimodal model: assumes a single, positively skewed mode. Well-suited to latency or cost data that are strictly positive and unimodal; supports censoring (timeout runs) by truncating the likelihood and updating the posterior accordingly.                                                         |
| **Bayesian Plackett-Luce Ranking Model** (Calvo et al., 2019)      | 2 +              | Multiple               | Independent rankings           | Yes – worth parameters follow a Dirichlet prior; ranking likelihood is PL                                              | Converts per-dataset performance tables into rankings and fits a PL model, yielding posterior “worth” scores for every algorithm. Delivers probabilities such as *P(alg i is best)* or pairwise win probabilities, and naturally handles any number of algorithms, ties, or missing results.                                |
| **Bayesian Wilcoxon Signed-Rank Test** (Benavoli et al., 2014)     | 2                | ≥ 1                    | Paired – independent **or** correlated                                         | Non-parametric (Dirichlet-process prior on the CDF of differences)                                                     | Replaces the deterministic ranking of classical Wilcoxon with a DP mixture, giving a full posterior for the median paired difference. Provides *P(A $\approx$ B)* as well as loss-minimising decisions, while retaining Wilcoxon’s robustness to non-normal, heavy-tailed, or ordinal data.                                 |




## Getting Started

This project is based on [CmdStanPy](https://mc-stan.org/cmdstanpy/), and is currently supported only on Linux systems.
To run it on Windows, you'll need to [set up a WSL (Windows Subsystem for Linux) development environment](https://learn.microsoft.com/en-us/windows/wsl/setup/environment).

### 1. Install CmdStanPy

Follow the official [CmdStanPy installation instructions](https://mc-stan.org/cmdstanpy/installation.html#conda-install-cmdstanpy-cmdstan-c-toolchain).
It is strongly recommended to use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python)  for managing the environment:

```bash
conda create -n stan -c conda-forge cmdstanpy
conda activate stan
conda info -e
```

### 2. Install Required Packages

Install additional dependencies:

```bash
conda install -c conda-forge arviz seaborn
```

### 3. Set Project Path

Make sure to add your project directory to the `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/your/project:$PYTHONPATH"
```

### 4. Run an Example

To test that everything is working, run the example script:

```bash
python run examples/run_beta_binomial_test.py
```


## Example Use
Example of Bayesian Independent T-Test with synthetic data.
```python
import numpy as np
from utils.helper import load_results
from utils.plotting import plot_densities
from bayesian_test.BayesianIndependentTTest import BayesianIndependentTTest


def example_01(seed: int) -> None:
    n_instances = 100  # Number of instances to generate for each sample

    # Generate synthetic data for y1 and y2 from normal distributions
    y1 = np.random.normal(loc=0.1, scale=0.4, size=n_instances)
    y2 = np.random.normal(loc=0.3, scale=0.6, size=n_instances)

    # Plot the density distribution for y1 and y2
    data = np.stack((y1, y2), axis=1)
    plot_densities(data, algorithm_labels=['Alg1', 'Alg2'], show_plt=True, alpha=1.0)

    # Initialize the Bayesian Independent T-Test model
    model = BayesianIndependentTTest(y1=y1, y2=y2, rope=(-0.01, 0.01), seed=seed)

    # Fit the model with specified number of sampling iterations, warmup iterations and chains
    model.fit(iter_sampling=50000, iter_warmup=1000, chains=4)

    # Define plot parameters for posterior pdf plot
    plot_parameter = dict(
        plt_imp_param=True,
        # Whether only the mu difference, sigma difference, effect size, mu1 and mu2 should be plotted. Default is False
        hpdi_prob=0.99,  # Highest Posterior Density Interval (HPDI) probability. Default is 0.99.
        plot_type='kde',  # Type of plot ('kde' or 'hist'). Default is 'kde'.
        plt_rope=True,  # Whether to plot the Region of Practical Equivalence (ROPE). Default is True.
        plt_rope_text=True,  # Whether to display text for the ROPE. Default is True.
        plt_within_rope=True,  # Whether to display the percentage of samples within the ROPE. Default is True.
        plt_mean=True,  # Whether to plot the mean of the samples. Default is False.
        plt_mean_text=True,  # Whether to display text for the mean. Default is False.
        plt_hpdi=True,  # Whether to plot the HPDI. Default is True.
        plt_hpdi_text=True,  # Whether to display text for the HPDI. Default is True.
        plt_samples=True,  # Whether to plot the sample points. Default is True.
        plt_title=True,  # Whether to display the title. Default is True.
        alpha=0.5,  # Transparency level for plot elements. Default is 0.8.
        font_size=10,  # Font size for text annotations. Default is 12.
    )

    # Analyze the model and generate plots
    model.analyse(posterior_predictive_check=True, file_name='example01', plot=True, save=True, round_to=3,
                  directory_path='examples/results', **plot_parameter)


def main():
    seed = 42  # Set the random seed for reproducibility

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Run the example of a Bayesian Independent T-Test
    example_01(seed)

    # Load saved results from example01
    results = load_results(file_path='examples/results/bayesian_independent_t_test/250805_1824', file_name='example01')
    for key in ['difference_mean', 'difference_sigma', 'effect_size', 'mu1', 'mu2', 'sigma1', 'sigma2', 'nu']:
        print(f'{key}: {results[key]["posterior_probabilities"]}')


if __name__ == '__main__':
    main()  # Execute the main function

```
More examples can be found in the [examples folder]().

## Motivation
Performance comparisons between two or more machine learning algorithms are essential to determine the most effective model for a given task. Evaluating the actual performance distribution of these algorithms requires statistical assessments to ensure that the observed differences are not due to random variation but reflect real differences in performance. This evaluation is critical to making informed decisions about which algorithms should be used in real-world applications.

The null hypothesis significance test (NHST) is a common statistical method used in all scientific fields that rely on empirical observations. NHST often uses frequentist methods, which rely on the frequency or proportion of data to draw conclusions.  Common frequentist NHST methods include, for example, the correlated t-test ([Nadeau and Bengio, 2003](https://doi.org/10.1023/A:1024068626366)) or the Wilcoxon-Mann-Whitney test ([Demšar, 2006](https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf)) for non-parametric pairwise comparisons. These tests focus on the null hypothesis (H0) that the populations have the same mean and use p-values to determine statistical significance. 

### The Issue with Null Hypothesis Significance Testing
However, p-values are often misinterpreted, leading to incorrect conclusions ([Demšar, 2008](https://www.site.uottawa.ca/ICML08WS/papers/J_Demsar.pdf)). The American Statistical Association has also issued a statement pointing out the limitations and possible misuse of p-values ([Wasserstein and Lazar, 2016](https://doi.org/10.1080/00031305.2016.1154108)).  

Although the NHST is widely criticized, it remains a prerequisite for publication, with $p \leq 0.05$ often regarded as objective evidence of the quality of a method. In [Benavoli et al., 2017](https://www.jmlr.org/papers/volume18/16-305/16-305.pdf), section "2.1 NHST: The pitfalls of black and white thinking" discusses the pitfalls of this confidence and points out the significant problems that arise from this assumption:

1. **Black and white thinking**: Decisions based on $p \leq 0.05$ lead to a false binary distinction and ignore that statistical significance is not synonymous with practical significance ([Berger and Sellke, 1987](https://www2.stat.duke.edu/courses/Spring07/sta215/lec/BvP/BergSell1987.pdf)). Furthermore, two methods that are not statistically different are not necessarily equivalent.

2. **Misinterpretation of confidence levels**: Researchers often misinterpret confidence levels, thinking that a 90% confidence level means that one algorithm has a 90% probability of outperforming another. In reality, the p-value indicates the probability of observing the data, assuming that the null hypothesis is true, not the probability of the hypotheses themselves.

3. **Point-wise null hypotheses**: Null hypotheses are almost always false because no two algorithms are completely equivalent. The NHST can often reject the null hypothesis with sufficient data, even if the effect size is trivial.

4. **Confuse effect size and sample size**: The p-value confuses effect size and sample size. Larger sample sizes may reveal even trivial differences, while small sample sizes may not reveal significant differences.

5. **Ignoring magnitude and uncertainty**:  The NHST does not provide information about the magnitude of the effect or the uncertainty of its estimate. This can lead to the null hypothesis being rejected for trivial effects or high uncertainty.

6. **No information about the null hypothesis**: If the null hypothesis cannot be rejected, this provides no evidence in favor of it. The NHST does not allow any conclusions to be drawn about the probability that the null hypothesis is true.

7. **Arbitrary significance levels**: No principled method for determining the significance threshold (e.g. 0.05) exists. This threshold is arbitrary and does not necessarily reflect meaningful differences.

8. **Dependence on sampling intentions**: The sampling distribution used in NHST depends on the researcher's intentions (e.g., to collect a certain number of observations). This is challenging to formalize and is often ignored, affecting the p-value's validity.


### Bayesian Inference as a Solution
Recently, Bayesian inference has been proposed in the machine learning community as a better alternative to NHST. Unlike NHST, which tests a null hypothesis, Bayesian methods estimate relevant information about the power distribution, focusing on parameters representing power differences.

Bayesian inference addresses the actual questions of interest:
* **Is method A better than method B?**
* **How likely is it that method A is better?**
* **What is the probability that method A is better by more than 1%?**

These questions are about posterior probabilities, which are naturally given with Bayesian methods. Bayesian inference provides a more flexible, informative, and intuitive framework for comparing machine learning algorithms. ([Benavoli et al., 2017](https://www.jmlr.org/papers/volume18/16-305/16-305.pdf)) 

By focussing on posterior probabilities and incorporating prior knowledge, Bayesian methods provide a more precise and more nuanced understanding of differences in model performance.
The following explains why Bayesian inference is better:

1. **Direct probability statements**: Bayesian methods allow us to make direct probability statements about hypotheses. For example, we can calculate the probability that method A is better than method B given the observed data, which is impossible with NHST.

2. **Incorporation of prior information**: Bayesian inference considers prior information or assumptions about the parameters that can be updated with new data. This allows for a more comprehensive understanding of the uncertainty and variability of performance differences.

3. **Effect size estimation**: Bayesian methods provide a full posterior distribution of the effect size, not just a single p-value. This allows researchers to understand the range and distribution of possible effect sizes rather than making a binary decision based on an arbitrary threshold.

4. **Handling small sample sizes**: Bayesian inference can be more robust with small sample sizes because it combines prior information with observed data to make more informed estimates. This contrasts NHST, where small sample sizes can lead to misleading p-values.

5. **Avoidance of "black and white thinking "**: Bayesian methods avoid the "black and white thinking" associated with p-values in NHST. Rather than simply declaring results as significant or not significant, Bayesian inference provides a nuanced view of the evidence and shows the likelihood of different hypotheses being true. 

6. **More informative results**: Bayesian inference provides more prosperous, more informative results that can be used to answer practical questions about model performance directly. For example, the probability that method A outperforms method B by a certain percentage can be quantified, leading to actionable insights.


## Further Reading
### Recommended Books 
**An accessible approach to Bayesian Data Analysis:**
* Kuschke, J. (2014). Doing Bayesian data analysis: A tutorial with R, JAGS, and Stan. Second Edition, Academic Press is an imprint of Elsevier, ISBN: 9780124058880. [https://nyu-cdsc.github.io/learningr/assets/kruschke_bayesian_in_R.pdf](https://nyu-cdsc.github.io/learningr/assets/kruschke_bayesian_in_R.pdf)

**A deep dive into Bayesian Data Analysis:**
* Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., & Rubin, D.B. (2013). Bayesian Data Analysis, Third Edition. Chapman & Hall/CRC Texts in Statistical Science. Taylor & Francis. ISBN: 9781439840955. LCCN: 2013039507. [http://www.stat.columbia.edu/~gelman/book/BDA3.pdf](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf)


### Key Paper

**An easy introduction to Bayesian Analysis:**
* Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. _Journal of Machine Learning Research_, _18_(77), 1-36. [https://www.jmlr.org/papers/volume18/16-305/16-305.pdf](https://www.jmlr.org/papers/volume18/16-305/16-305.pdf)


### Helpful Videos

**A video series for an easy introduction to Bayesian Data Analysis:**
* rasmusab. (2017). Introduction to Bayesian data analysis - part 1: What is Bayes?. YouTube. [https://www.youtube.com/watch?v=3OJEae7Qb_o&list=PLYXGE3IMa0C4FwmZYwo5tqorv_VUM32XX&index=3&ab_channel=rasmusab](https://www.youtube.com/watch?v=3OJEae7Qb_o&list=PLYXGE3IMa0C4FwmZYwo5tqorv_VUM32XX&index=3&ab_channel=rasmusab)
* rasmusab. (2017). Introduction to Bayesian data analysis - part 2: Why use Bayes?. YouTube. [https://www.youtube.com/watch?v=mAUwjSo5TJE&list=PLYXGE3IMa0C4FwmZYwo5tqorv_VUM32XX&index=1&ab_channel=rasmusab](https://www.youtube.com/watch?v=mAUwjSo5TJE&list=PLYXGE3IMa0C4FwmZYwo5tqorv_VUM32XX&index=1&ab_channel=rasmusab)
* rasmusab. (2017). Introduction to Bayesian data analysis - part 3: How to do Bayes?. YouTube. [https://www.youtube.com/watch?v=Ie-6H_r7I5A&list=PLYXGE3IMa0C4FwmZYwo5tqorv_VUM32XX&index=3&ab_channel=rasmusab](https://www.youtube.com/watch?v=Ie-6H_r7I5A&list=PLYXGE3IMa0C4FwmZYwo5tqorv_VUM32XX&index=3&ab_channel=rasmusab)

### Tutorials
* [A small tutorial from the scmamp repository.](https://github.com/b0rxa/scmamp/tree/master/vignettes)

# References
* Benavoli, A., Corani, G., Mangili, F., Zaffalon, M., & Ruggeri, F. (2014, June). A Bayesian Wilcoxon signed-rank test
  based on the Dirichlet process. In_International conference on machine learning_(pp. 1026-1034). PMLR. 
  [https://people.idsia.ch/~alessio/benavoli2014a.pdf](https://people.idsia.ch/~alessio/benavoli2014a.pdf)
* Benavoli, A., Corani, G., Demšar, J., & Zaffalon, M. (2017). Time for a change: a tutorial for comparing multiple 
  classifiers through Bayesian analysis. _Journal of Machine Learning Research_,_18_(77), 1-36. 
  [https://www.jmlr.org/papers/volume18/16-305/16-305.pdf](https://www.jmlr.org/papers/volume18/16-305/16-305.pdf)
* Calvo, B., & Santafé Rodrigo, G. (2016). scmamp: Statistical comparison of multiple algorithms in multiple problems. 
  The R Journal, Vol. 8/1, Aug. 2016.[https://journal.r-project.org/archive/2016/RJ-2016-017/RJ-2016-017.pdf](https://journal.r-project.org/archive/2016/RJ-2016-017/RJ-2016-017.pdf)
* Calvo, B., Shir, O. M., Ceberio, J., Doerr, C., Wang, H., Bäck, T., & Lozano, J. A. (2019, July). Bayesian performance
  analysis for black-box optimization benchmarking. In_Proceedings of the Genetic and Evolutionary Computation
  Conference Companion_(pp. 1789-1797). [https://dl.acm.org/doi/pdf/10.1145/3319619.3326888](https://dl.acm.org/doi/pdf/10.1145/3319619.3326888)
* Corani, G., & Benavoli, A. (2015). A Bayesian approach for comparing cross-validated algorithms on multiple data sets.
  _Machine Learning_,_100_(2), 285-304.[https://doi.org/10.1007/s10994-015-5486-z](https://doi.org/10.1007/s10994-015-5486-z)
* Corani, G., Benavoli, A., Demšar, J., Mangili, F., & Zaffalon, M. (2017). Statistical comparison of classifiers
  through Bayesian hierarchical modelling._Machine Learning_,_106_, 1817-1837.
  [https://doi.org/10.1007/s10994-017-5641-9](https://doi.org/10.1007/s10994-017-5641-9)
* Kruschke, J. K. (2013). Bayesian estimation supersedes the t test._Journal of Experimental Psychology: General_,_142_(
  2), 573. [https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf](https://jkkweb.sitehost.iu.edu/articles/Kruschke2013JEPG.pdf)
* von Pilchau, W. P., Pätzel, D., Stein, A., & Hähner, J. (2023, June). Deep Q-Network Updates for the Full Action-Space 
  Utilizing Synthetic Experiences. In 2023 International Joint Conference on Neural Networks (IJCNN) (pp. 1-9). IEEE.
  [https://doi.org/10.1007/s10994-015-5486-z](https://doi.org/10.1007/s10994-015-5486-z)

## Citing the Project
```
@misc{kemper2025,
  author = {Kemper, Neele},
  title = {bayesian-model-comparison},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NeeleKemper/bayesian-model-comparison}},
}
```