## Overview

Bayesian inference of T cell allelic inclusion rates from emulsion-barcoding single-cell sequencing data. 

* [Model](https://github.com/JasonACarter/Allelic_inclusion/Model)
  * [Derivations](https://github.com/JasonACarter/Allelic_inclusion/Model/Derivations): iPython notebooks detailing step-by-step derivation of likelihood function and full Bayesian framework
  * [Inclusion.py](https://github.com/JasonACarter/Allelic_inclusion/Model/Inclusion.py): Python implementation of our Bayesian inference model
* [Data](https://github.com/JasonACarter/Allelic_inclusion/Data)
  * Distribution of α and β TCR chains per droplet for each individual (Fig. 1)
  * TCR seqeunces for plate-based sequencing approach (Fig. 2)
  * Dataframes containing all αβ, ααβ, and αββ TCR sequence sets used in our study (Figs. 3 & 4)
  * Version of [VDJdb](https://vdjdb.cdr3.net) used in Fig. 4
* [Figures](https://github.com/JasonACarter/Allelic_inclusion/Figures)
  * iPython notebooks containing code necessary to generate each main manuscript figure
    
## Requirements
**Python version:** Python 3 (tested with 3.6.8)

**Dependencies:** Numpy (tested with version 1.16.2), Pandas (0.23.4), scipy (1.2.1), emcee (2.2.1), numba (0.42.0), matplotlib (2.2.3), seaborn (0.9.0)


## Download

The allelic inclusion inference model can be downloaded using:

```git clone https://github.com/JasonACarter/Allelic_inclusion/Model/Inclusion.py```

## Use 

### Input
Array(s) containing the number of droplets observed to contain *n* unique α and/or β T cell receptor (TCR) sequences. If no experimental distributions are provided, simulated distributions will be generated and used. 

### Output

Maximum *a posteriori* (MAP) estimates and credible intervals for each of the 4 model parameters. 

### Parameters

* **Inference of experimental distributions**
  * **alpha:** Specify array containing experimental α TCR chain distribution to fit (default=None)
  * **beta:** Specify array containing experimental β TCR chain distribution to fit (default=None)
  
* **Inference of simulated distributions**
  * **chains:** If no experimental distributions are specified, should α (alpha), β (beta), or αβ (pairs) distributions be simulated and fit (default=pairs)
  * **la_real:** True loading rate (λ) to use for simulated distributions. As the loading rate refer to T cells, the same rate is used for both α and β distributions. 
  * **ga_real:** True error loading rate (γ) to use for simulated distributions. As the error loading rate refer to T cells, the same rate is used for both α and β distributions. 
  * **f_real_(a)(b):** True allelic inclusion rates to use for simulated αβ distributions
  * **s_real_(a)(b):** True TCR chain dropout rates to use for simulated αβ distributions

* **MCMC parameters** (see [emcee](https://emcee.readthedocs.io/en/stable/))
  * **nwalkers:** Number of [walkers](https://emcee.readthedocs.io/en/stable/user/faq/#what-are-walkers)
  * **steps:** MCMC chain length
  * **burn:** Number of steps for burn-in
  * **N_random_starts** Number of random starts to find MAP
  * For these parameters, larger values will provide more accurates estimates at the exspence of run-time
  
 * **Output**  
    * **calculate_posterior:** Whether to calculate posterior distributions (default=1). If 0, will not run MCMC sampling and estimates may therefore not be accurate.  
    * **print_estimates:** Whether to print MAP estimates with credible intervals for each parameter (default=1).
    * **plot_dist:** Whether to plot posterior distributions for each parameter (default=0).
    * **interval:** Specify size of credible interval for print_estimates (default=68).
 
## Example usage 

### Simulated distributions

Infer parameter values from simulated αβ distributions with user-specified parameter values: 
```Python
import Inclusion
Inclusion(chains='pairs', la_real=0.08,ga_real=4.15,f_real_a=0.083,f_real_b=0.043,s_real_a=0.53,s_real_b=0.39)
```
Which should output: 

```
No experimental distributions specified. Running paired simulation. 
Simulated alpha distribution with the following parameters: la:0.08, ga:4.15, f:0.083, s:0.53 
Simulated beta distribution with the following parameters: la:0.08, ga:4.15, f:0.043, s:0.39 

MAP Estimates:
la: 0.537 (0.256,0.856) vs. 0.08
ga: 1.284 (0.234,2.482) vs. 4.15
f_a: 0.537 (0.105,0.829) vs. 0.083
s_a: 0.969 (0.354,0.843) vs. 0.53
f_b: 0.530 (0.178,0.831) vs. 0.043
s_b: 0.544 (0.184,0.740) vs. 0.39
```
Parameter values are given as "MAP (68% credible interval) vs. true value used in simulation."

### Experimental distributions

```Python
counts_alpha=[829988, 812208, 75516, 8203, 1645, 593, 232, 136, 70, 46, 24, 12]
counts_beta=[605727, 1015629, 92206, 11011, 2359, 808, 397, 238, 122, 76, 55, 25]
Inclusion(alpha=counts_alpha,beta=counts_beta)
```

Which should output: 

```
Experimental paired distribution specificed. 

MAP Estimates:
la: 0.077 (0.069,0.093)
ga: 4.144 (4.134,4.155)
f_a: 0.086 (0.079,0.092)
s_a: 0.530 (0.530,0.531)
f_b: 0.042 (0.035,0.048)
s_b: 0.387 (0.386,0.387)
```
Parameter values are again given as "MAP (68% credible interval)", but no true values are unknown for experimental distributions. 

## Citation

For inference model and allelic inclusion TCR sequences:

* Carter, J.A., Preall, J.B., and Atwal, G.S. (2019) "Bayesian Inference of Allelic Inclusion Rates in the Human T Cell Receptor Repertoire." Cell Systems 9 doi:https://doi.org/10.1016/j.cels.2019.09.006

For paired αβ TCR sequences: 

* Carter *et al.* (2019) "Single T Cell Sequencing Demonstrates the Functional Role of αβ TCR Pairing in Cell Lineage and Antigen Specificity." Front. Immunol. 10:1516. doi:https://doi.org/10.3389/fimmu.2019.01516

* Grigaityte *et al.* (2017) "Single-cell sequencing reveals αβ chain pairing shapes the T cell repertoire." bioRxiv 213462; doi:https://doi.org/10.1101/213462

## Contact

For questions or bugs, please contact Jason Carter (jacarter@cshl.edu) or Mickey Atwal (atwal@cshl.edu). 

## License

Copyright (c) 2019, Jason Carter, Jonathan Preall, and Gurinder Atwal. 
All rights reserved.

Permission to use, copy, modify and distribute any part of this program for educational, research and non-profit purposes, by non-profit institutions only, without fee, and without a written agreement is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in all copies.

Those desiring to incorporate this work into commercial products or use for commercial purposes should contact Gurinder Atwal (atwal@cshl.edu). 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

