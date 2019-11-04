#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import factorial as fac
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import optimize
from scipy.stats import rv_discrete
import warnings
from numba import jit
import emcee

class Inclusion(object):
    def __init__(self,chains="Pairs",calculate_posterior=1,print_estimates=1,
                 plot_dist=0,plot_posterior=0,interval=68,
                 alpha=None,beta=None,
                 la_real=0.08,ga_real=4.15,N=int(1e6),
                 f_real_a=0.083,s_real_a=0.53,
                 f_real_b=0.043,s_real_b=0.39,
                 nwalkers=200,steps=1000,burn=100,
                 N_random_starts=50,manual=0):
        self.chains,self.bayes,self.print_estimates=[chains,calculate_posterior,print_estimates]
        self.plot,self.interval,self.plot_dist=[plot_posterior,interval,plot_dist]
        self.alpha,self.beta=[alpha,beta]
        self.la_real,self.ga_real,self.N=N=[la_real,ga_real,int(N)]
        self.f_real_a,self.s_real_a=[f_real_a,s_real_a]
        self.f_real_b,self.s_real_b=[f_real_b,s_real_b]
        self.nwalkers,self.steps,self.burn=[nwalkers,steps,burn]
        self.N_random_starts,self.manual=[N_random_starts,manual]
        if self.bayes!=1 and self.plot==1:
            print('Plots require MCMC sampling of posterior. Setting calculate_posterior equal to 1. \n')
            self.bayes=1
        if self.manual==0:
            self.run_type()
            self.run_all()
    
    def run_type(self):
        if self.alpha is None and self.beta is None:
            if self.chains.lower() in ["pairs","paired","p","ab"]:
                self.use_chains="paired"
            elif self.chains.lower() in ["alpha","a"]:
                self.use_chains="alpha"
            elif self.chains.lower() in ["beta","b"]:
                self.use_chains="beta"
            if self.print_estimates==1:
                print(f'No experimental distributions specified. Running {self.use_chains} simulation. \n')
            self.run='Simulate'
        else:
            if self.alpha is not None and self.beta is None:
                self.use_chains='alpha'
            elif self.alpha is None and self.beta is not None:
                self.use_chains='beta'
            else:
                self.use_chains='paired'
            if self.print_estimates==1:
                print(f'Experimental {self.use_chains} distribution specificed. \n')
            self.run='Experimental'
            
    def run_all(self):
        if self.run=='Simulate':
            if self.use_chains=='alpha' or self.use_chains=='paired':
                self.alpha=self.simulate('alpha')
            if self.use_chains=='beta' or self.use_chains=='paired':
                self.beta=self.simulate('beta')
        if self.bayes==1:
            self.sampler=self.MCMC()
            self.opt=self.find_MAP()
            if self.plot==1:
                self.posterior_plot()
        else:
            self.opt=self.find_MAP()
        if self.print_estimates==1:
            self.print_map_estimates()
        if self.plot_dist==1:
            self.dist_plot()
        self.return_map_estimates()
            
    def print_map_estimates(self):
        if self.use_chains=='paired':
            labels=['la','ga','f_a','s_a','f_b','s_b']
            reals=[self.la_real,self.ga_real,self.f_real_a,self.s_real_a,self.f_real_b,self.s_real_b]
        else:
            labels=['la','ga','f','s']
            if self.use_chains=='alpha':
                reals=[self.la_real,self.ga_real,self.f_real_a,self.s_real_a]
            else:
                reals=[self.la_real,self.ga_real,self.f_real_b,self.s_real_b]
      
        print('MAP Estimates:')
        if self.bayes==0:
            for i in range(self.opt.shape[1]-2):
                if self.run=='Experimental': 
                    print(f'{labels[i]}: {float(self.opt[labels[i]].iloc[0]):.3f}')
                else:
                    print(f'{labels[i]}: {float(self.opt[labels[i]].iloc[0]):.3f} vs. {reals[i]}')
        else:
            for i in range(self.sampler.flatchain.shape[1]):
                sample=np.sort(self.sampler.flatchain[:,i])
                cutoff=int(len(sample)*((1-self.interval/100)/2))
                low=sample[cutoff]
                high=sample[-cutoff]
                if self.run=='Experimental':
                    print(f'{labels[i]}: {float(self.opt[labels[i]].iloc[0]):.3f} ({low:.3f},{high:.3f})')
                else:
                    print(f'{labels[i]}: {float(self.opt[labels[i]].iloc[0]):.3f} ({low:.3f},{high:.3f}) vs. {reals[i]}')
                    
    def return_map_estimates(self):
        if self.use_chains=='paired':
            labels=['la','ga','f_a','s_a','f_b','s_b']
            reals=[self.la_real,self.ga_real,self.f_real_a,self.s_real_a,self.f_real_b,self.s_real_b]
        else:
            labels=['la','ga','f','s']
            if self.use_chains=='alpha':
                reals=[self.la_real,self.ga_real,self.f_real_a,self.s_real_a]
            else:
                reals=[self.la_real,self.ga_real,self.f_real_b,self.s_real_b]
      
        self.return_map=[self.opt[labels[i]].iloc[0] for i in range(self.opt.shape[1]-2)]
        if self.bayes==1:
            self.return_map_interval_lower=np.zeros(self.sampler.flatchain.shape[1])
            self.return_map_interval_upper=np.zeros(self.sampler.flatchain.shape[1])
            for i in range(self.sampler.flatchain.shape[1]):
                sample=np.sort(self.sampler.flatchain[:,i])
                cutoff=int(len(sample)*((1-self.interval/100)/2))
                self.return_map_interval_lower[i]=sample[cutoff]
                self.return_map_interval_upper[i]=sample[-cutoff]
        
    def posterior_plot(self):
        plt.figure(figsize=(20,30))
        labels=['la','ga','f_a','s_a','f_b','s_b']
        reals=[self.la_real,self.ga_real,self.f_real_a,self.s_real_a,self.f_real_b,self.s_real_b]
        for i in range(self.sampler.flatchain.shape[1]):
            plt.subplot(self.sampler.flatchain.shape[1],1, i+1)
            p=sns.kdeplot(self.sampler.flatchain[:,i],shade=True,color='Gray',label=f'{labels[i]} Posterior',bw=.01)
            x,y = p.get_lines()[0].get_data()
            if self.run!='Experimental':
                plt.plot([reals[i],reals[i]],[0,int(1e6)],'r',label='Real')
            plt.plot([float(self.opt.iloc[0][labels[i]]),float(self.opt.iloc[0][labels[i]])],[0,int(1e6)],'k',label='MAP')
            plt.ylim([0,int(np.max(y))+1])
            interval=99
            sample=np.sort(self.sampler.flatchain[:,i])
            cutoff=int(len(sample)*((1-interval/100)/2))
            low=sample[cutoff]
            high=sample[-cutoff]
            mean=float(self.opt.iloc[0][labels[i]])
            rang=np.max((mean-low,high-low))
            plt.xlim([np.max((0,mean-rang)),mean+rang])
            plt.legend()
            
    def dist_plot(self):
        def px(counts,la,f_a,ga,s_a):
            n_max=20
            m_max=10
            pre=(np.exp(-la))/((1-np.exp(-la))*scipy.special.zeta(ga))
            loadings=[self.p_loading_convolution(chains,la,ga) for chains in range(n_max+m_max+1)]
            inclusions_a=self.inclusion_probabilities(n_max,m_max,f_a)
            return [self.p_observing_x_chains(i,la,f_a,s_a,ga,loadings,inclusions_a,pre,n_max,m_max) for i in range(0,len(counts))]
       
        def plots(df,counts,rowname_f,rowname_s,names):
            theory=px(counts,float(df.iloc[0]['la']),float(df.iloc[0][rowname_f]),float(df.iloc[0]['ga']),float(df.iloc[0][rowname_s]))
            if self.run=='Experimental':
                label='Experiment'
            else:
                label='Simulation'
            plt.bar(range(len(counts)),counts/sum(counts),width=0.3,label=label,color='Black')
            plt.bar(np.arange(len(counts))+0.3,theory,width=0.3,label='Theory',color='darkgray')
            plt.xticks(np.arange(len(counts))+0.15,np.arange(len(counts)))  
            plt.xlabel(f'{names} chains per droplet',fontsize=15)
            plt.ylabel('probability',fontsize=15)
            plt.yscale('log')
            plt.legend(fontsize=15)
            plt.show()
            plt.close()
    
        df=self.opt
        names=['Alpha','Beta']
        rowname_f=['f_a','f_b']
        rowname_s=['s_a','s_b']
        if self.use_chains=='paired':
            for j,counts in enumerate([self.alpha,self.beta]):
                plots(df,counts,rowname_f[j],rowname_s[j],names[j])
        elif self.use_chains=='alpha':
            plots(df,self.alpha,'f','s',names[0])
        elif self.use_chains=='beta':
            plots(df,self.beta,'f','s',names[1])
            
    def find_MAP(self):
        def optimization():
            warnings.simplefilter(action='ignore')
            method=['L-BFGS-B']
            N_methods=len(method)
            if self.use_chains=='paired':
                sample_size=self.alpha+self.beta
                length=6
            elif self.use_chains=='alpha':
                sample_size=self.alpha
                length=4
            elif self.use_chains=='beta':
                sample_size=self.beta
                length=4
                
            for starting in range(self.N_random_starts):
                if self.bayes==1:
                    x0=[np.mean(self.sampler.flatchain[:,i]) for i in range(self.sampler.flatchain.shape[1])]
                    x0=np.absolute(x0+np.random.normal(0,.1,len(x0)))
                else:
                    if length==6:
                        x0=[np.random.random(1),sum(np.random.random(8)),np.random.random(1),np.random.random(1),np.random.random(1),np.random.random(1)]
                    elif length==4:
                        x0=[np.random.random(1),sum(np.random.random(8)),np.random.random(1),np.random.random(1)]
                if len(x0)==6:
                    bounds=((0,1),(1.01,20),(0,1),(0,1),(0.01,1),(0,1))
                else:
                    bounds=((0,1),(1.01,20),(0,1),(0,1))
                data=np.empty((N_methods,length+2),dtype=object)
                for it,methods in enumerate(method):
                    neg_LL=lambda *args: -self.posterior(*args)/np.sum(sample_size)
                    try:
                        minimum=optimize.minimize(neg_LL,x0,method=methods,bounds=bounds)
                    except:
                        minimum=optimize.minimize(neg_LL,x0,method=methods)
                    data[it]=np.hstack((minimum.x,minimum.fun,methods)) 
                if starting==0:
                    df=data
                else:
                    df=np.vstack((df,data))
            if length==6:
                df=pd.DataFrame(df,columns=['la','ga','f_a','s_a','f_b','s_b','energy','method'])
            elif length==4:
                df=pd.DataFrame(df,columns=['la','ga','f','s','energy','method'])
            df=df.sort_values(by=['energy'])
            df=df[df['energy'].astype(float)>0]
            df=df[df.la.astype(float)>0.001]
            return df
        return optimization()
                
    def MCMC(self):
        if self.use_chains=='paired':
            self.ndim=6
        else:
            self.ndim=4
        p0 = np.random.rand(self.ndim * self.nwalkers).reshape((self.nwalkers, self.ndim))
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.posterior)
        pos, prob, state = sampler.run_mcmc(p0, self.burn)
        sampler.reset()
        sampler.run_mcmc(pos, self.steps)
        return sampler
    
    def posterior(self,theta):
        prior_prob=self.prior(theta)
        if np.isfinite(prior_prob):
            return self.prior(theta) + self.log_likelihood(theta)
        else:
            return self.prior(theta)
    
    def p_poisson(self,x,la):
            return ((la**x))/((fac(x)))
    def p_binomial_inclusion(self,n,m,f):
        return (fac(n+m)*(f**m)*((1-f)**n))/(fac(n)*fac(m))
    def inclusion_probabilities(self,n_max,m_max,f):
        inclusions=np.zeros((n_max,m_max))
        for n in range(n_max):
            for m in range(m_max):
                inclusions[n,m]=self.p_binomial_inclusion(n,m,f)
        return inclusions
    def p_binomial_dropout(self,total_chains,x,s):
        try:
            return ((fac(total_chains))*(s**(total_chains-x)))/(fac(total_chains-x))
        except:
            return 0
    def p_loading_power(self,x,ga):
        return ((x)**(-ga))
    def p_loading_convolution(self,x,la,ga):
        prop=0
        for k in range(1,x+1):
            prop=prop+(self.p_poisson(k,la)*self.p_loading_power(x+1-k,ga))
        return prop
    def p_observing_x_chains(self,x,la,f,s,ga,loading,inclusion,pre,n_max,m_max):
        prefactor=pre*(((1-s)**x)/(fac(x)))
        dropout=[self.p_binomial_dropout(chains,x,s) for chains in range(n_max+2*m_max+1)]
        total=0
        for n in range(n_max):
            for m in range(m_max):
                if 2*m+n>=x and (m+n)!=0:
                    total=total+(loading[n+m]*inclusion[n,m]*dropout[2*m+n])
        return prefactor*total
                
    def log_likelihood(self,theta): 

        if len(theta)==4:
            la,ga,f_a,s_a=theta
            if self.use_chains=='alpha':
                counts_alpha=self.alpha
            else:
                counts_alpha=self.beta    
        elif len(theta)==6:
            la,ga,f_a,s_a,f_b,s_b=theta
            counts_alpha=self.alpha
            counts_beta=self.beta
        n_max=20
        m_max=10
        pre=(np.exp(-la))/((1-np.exp(-la))*scipy.special.zeta(ga))
        loadings=[self.p_loading_convolution(chains,la,ga) for chains in range(n_max+m_max+1)]
        inclusions_a=self.inclusion_probabilities(n_max,m_max,f_a)
        alpha=np.sum([counts_alpha[i]*np.log(self.p_observing_x_chains(i,la,f_a,s_a,ga,loadings,inclusions_a,pre,n_max,m_max)) for i in range(0,len(counts_alpha))])
        if len(theta)==6:
            inclusions_b=self.inclusion_probabilities(n_max,m_max,f_b)
            beta=np.sum([counts_beta[i]*np.log(self.p_observing_x_chains(i,la,f_b,s_b,ga,loadings,inclusions_b,pre,n_max,m_max)) for i in range(0,len(counts_beta))])
            return alpha+beta
        else:
            return alpha
    
    def prior(self,theta):
        if len(theta)==6:
            la,ga,f_a,s_a,f_b,s_b=theta
        elif len(theta)==4:
            la,ga,f_a,s_a=theta
            f_b,s_b=[0.5,0.5]
        if 0<=la<=1 and 0<=f_a<=1 and 0<=s_a<=1 and 0<=f_b<=1 and 0<=s_b<=1 and 1<ga<=20:
            return np.log(1)
        else: 
            return -np.inf
        
    def simulate(self,chain):
        la,ga,N=[self.la_real,self.ga_real,self.N]
        if chain=='alpha':
            f,s=[self.f_real_a,self.s_real_a]
        elif chain=='beta':
            f,s=[self.f_real_b,self.s_real_b]
        
        if self.print_estimates==1:
            print(f'Simulated {chain} distribution with the following parameters: la:{la}, ga:{ga}, f:{f}, s:{s} \n')
        
        def counts_from_x(x):
            counts_max=int(np.max(x))       
            counts=np.array([np.sum(1*x==i) for i in range(counts_max+1)])
            return counts
        
        @jit
        def loop(N,total_cells,allelic,chains,f_real,s_real,x,counts,counts_m):
            for t in range(N): #iteratre through N droplets
                cells_per_drop=total_cells[t] #cells per droplet
                random_allelic=allelic[counts:counts+cells_per_drop] #Assign random number (between 0 and 1) to each droplet
                counts=counts+cells_per_drop
                m=np.sum(1*(random_allelic<f_real))   #Determine number of allelic inclusion cells in droplet
                random_chains=chains[counts_m:counts_m+m+cells_per_drop] #Assign random number to each chain in droplet
                counts_m=counts_m+m+cells_per_drop
                x[t]=int(np.sum(1*(random_chains>s_real))) #Determine number of observed chains per droplet
            return counts_from_x(x) #Return distribution of chains observed per droplet
    
        def run_simulation(N,la,ga,f,s):
            ztp_distribution=[(np.exp(-la)*la**x)/((1-np.exp(-la))*fac(x)) for x in range(1,50)] #zero-truncated poissson distribution
            lambdas=rv_discrete(values=(range(1,50),ztp_distribution)).rvs(size=int(N)) #Sample from ZTP
            power_distribution=[(x**-ga)/(scipy.special.zeta(ga)) for x in range(1,50)] #Power law distribution
            powers = (rv_discrete(values=(range(1,50),power_distribution)).rvs(size=int(N)))-1 #Sample from power law
            total_cells=lambdas+powers #Number of cells per droplet
            allelic=np.random.rand(np.sum(total_cells)) #Random number to determine whether a given cell is allelic inclusion cell
            chains=np.random.rand(int(2*np.sum(total_cells))) #Random number to determine whether a given chain is observed
            counts=loop(N,total_cells,allelic,chains,f,s,x=np.zeros(N),counts=0,counts_m=0)
            return counts[:12]
        return run_simulation(N,la,ga,f,s)