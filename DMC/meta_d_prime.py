import numpy as np
from scipy.stats import norm
from scipy.optimize import Bounds, LinearConstraint, minimize, SR1, NonlinearConstraint
import matplotlib.pyplot as plt
import random
from functools import partial
from sklearn.metrics import brier_score_loss


class Meta_d_prime(object):
    def __init__(self) -> None:
        self.loss_record = []

    def trials2counts(self, stimID, response, rating, nRatings, padCells = 1, padAmount = None):

        """
        function [nR_S1, nR_S2] = trials2counts(stimID, response, rating, nRatings, padCells, padAmount)

        % Given data from an experiment where an observer discriminates between two
        % stimulus alternatives on every trial and provides confidence ratings,
        % converts trial by trial experimental information for N trials into response 
        % counts.
        %
        % INPUTS
        % stimID:   1xN vector. stimID(i) = 0 --> stimulus on i'th trial was S1.
        %                       stimID(i) = 1 --> stimulus on i'th trial was S2.
        %
        % response: 1xN vector. response(i) = 0 --> response on i'th trial was "S1".
        %                       response(i) = 1 --> response on i'th trial was "S2".
        %
        % rating:   1xN vector. rating(i) = X --> rating on i'th trial was X.
        %                       X must be in the range 1 <= X <= nRatings.
        %
        % N.B. all trials where stimID is not 0 or 1, response is not 0 or 1, or
        % rating is not in the range [1, nRatings], are omitted from the response
        % count.
        %
        % nRatings: total # of available subjective ratings available for the
        %           subject. e.g. if subject can rate confidence on a scale of 1-4,
        %           then nRatings = 4
        %
        % optional inputs
        %
        % padCells: if set to 1, each response count in the output has the value of
        %           padAmount added to it. Padding cells is desirable if trial counts 
        %           of 0 interfere with model fitting.
        %           if set to 0, trial counts are not manipulated and 0s may be
        %           present in the response count output.
        %           default value for padCells is 0.
        %
        % padAmount: the value to add to each response count if padCells is set to 1.
        %            default value is 1/(2*nRatings)
        %
        %
        % OUTPUTS
        % nR_S1, nR_S2
        % these are vectors containing the total number of responses in
        % each response category, conditional on presentation of S1 and S2.
        %
        % e.g. if nR_S1 = [100 50 20 10 5 1], then when stimulus S1 was
        % presented, the subject had the following response counts:
        % responded S1, rating=3 : 100 times
        % responded S1, rating=2 : 50 times
        % responded S1, rating=1 : 20 times
        % responded S2, rating=1 : 10 times
        % responded S2, rating=2 : 5 times
        % responded S2, rating=3 : 1 time
        %
        % The ordering of response / rating counts for S2 should be the same as it
        % is for S1. e.g. if nR_S2 = [3 7 8 12 27 89], then when stimulus S2 was
        % presented, the subject had the following response counts:
        % responded S1, rating=3 : 3 times
        % responded S1, rating=2 : 7 times
        % responded S1, rating=1 : 8 times
        % responded S2, rating=1 : 12 times
        % responded S2, rating=2 : 27 times
        % responded S2, rating=3 : 89 times
        """

        for i in range(len(stimID)):
            if stimID[i] == "yes":
                stimID[i] = 1
            if stimID[i] == "no":
                stimID[i] = 0
            if response[i] == "yes":
                response[i] = 1
            if response[i] == "no":
                response[i] = 0

        ''' sort inputs '''
        # check for valid inputs
        if not ( len(stimID) == len(response)) and (len(stimID) == len(rating)):
            raise('stimID, response, and rating input vectors must have the same lengths')
        
        ''' filter bad trials '''
        tempstim = []
        tempresp = []
        tempratg = []
        for s,rp,rt in zip(stimID,response,rating):
            if (s == 0 or s == 1) and (rp == 0 or rp == 1) and (rt >=1 and rt <= nRatings):
                tempstim.append(s)
                tempresp.append(rp)
                tempratg.append(rt)
        stimID = tempstim
        response = tempresp
        rating = tempratg
        
        ''' set input defaults '''
        if padAmount == None:
            padAmount = 1/(2*nRatings)
            
            
        
        ''' compute response counts '''
        nR_S1 = []
        nR_S2 = []

        # S1 responses
        for r in range(nRatings,0,-1):
            cs1, cs2 = 0,0
            for s,rp,rt in zip(stimID, response, rating):
                if s==0 and rp==0 and rt==r:
                    cs1 += 1
                if s==1 and rp==0 and rt==r:
                    cs2 += 1
            nR_S1.append(cs1)
            nR_S2.append(cs2)
        
        # S2 responses
        for r in range(1,nRatings+1,1):
            cs1, cs2 = 0,0
            for s,rp,rt in zip(stimID, response, rating):
                if s==0 and rp==1 and rt==r:
                    cs1 += 1
                if s==1 and rp==1 and rt==r:
                    cs2 += 1
            nR_S1.append(cs1)
            nR_S2.append(cs2)
        
        # pad response counts to avoid zeros
        if padCells:
            nR_S1 = [n+padAmount for n in nR_S1]
            nR_S2 = [n+padAmount for n in nR_S2]
            
        return nR_S1, nR_S2
    
    def __fit_meta_d_logL(self, parameters, inputObj):
        # if any(np.isnan(parameters)) or any(np.isinf(parameters)):
        #     print("parameters nan inf!!!!!!!!!")
        #     return 1e+300


        

        #print("lossx: ", parameters)
        meta_d1 = parameters[0]
        t2c1    = parameters[1:]
        nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv, beta, p = inputObj
        
        # define mean and SD of S1 and S2 distributions
        S1mu = -meta_d1/2
        S1sd = 1
        S2mu = meta_d1/2
        S2sd = S1sd/s

        # adjust so that the type 1 criterion is set at 0
        # (this is just to work with optimization toolbox constraints...
        #  to simplify defining the upper and lower bounds of type 2 criteria)
        S1mu = S1mu - eval(constant_criterion)
        S2mu = S2mu - eval(constant_criterion)

        t1c1 = 0

        # set up MLE analysis
        # get type 2 response counts
        # S1 responses
        # nC_rS1 = [nR_S1[i] for i in range(nRatings)]
        # nI_rS1 = [nR_S2[i] for i in range(nRatings)]
        # # S2 responses
        # nC_rS2 = [nR_S2[i+nRatings] for i in range(nRatings)]
        # nI_rS2 = [nR_S1[i+nRatings] for i in range(nRatings)]

        # get type 2 probabilities
        C_area_rS1 = fncdf(t1c1,S1mu,S1sd)
        I_area_rS1 = fncdf(t1c1,S2mu,S2sd)
        
        C_area_rS2 = 1-fncdf(t1c1,S2mu,S2sd)
        I_area_rS2 = 1-fncdf(t1c1,S1mu,S1sd)


        I_nR_rS2 = nR_S1[nRatings:]
        I_nR_rS1 = list(np.flip(nR_S2[0:nRatings],axis=0))
        
        C_nR_rS2 = nR_S2[nRatings:]
        C_nR_rS1 = list(np.flip(nR_S1[0:nRatings],axis=0))
        
        obs_FAR2_rS2 = [sum( I_nR_rS2[(i+1):] ) / sum(I_nR_rS2) for i in range(nRatings-1)]
        obs_HR2_rS2 = [sum( C_nR_rS2[(i+1):] ) / sum(C_nR_rS2) for i in range(nRatings-1)]
        obs_FAR2_rS1 = [sum( I_nR_rS1[(i+1):] ) / sum(I_nR_rS1) for i in range(nRatings-1)]
        obs_HR2_rS1 = [sum( C_nR_rS1[(i+1):] ) / sum(C_nR_rS1) for i in range(nRatings-1)]


        est_FAR2_rS2 = []
        est_HR2_rS2 = []
        
        est_FAR2_rS1 = []
        est_HR2_rS1 = []
        
        t2c1 = t2c1 + eval(constant_criterion)
        for i in range(nRatings-1):
            
            t2c1_lower = t2c1[(nRatings-1)-(i+1)]
            t2c1_upper = t2c1[(nRatings-1)+i]
                
            I_FAR_area_rS2 = 1-fncdf(t2c1_upper,S1mu,S1sd)
            C_HR_area_rS2  = 1-fncdf(t2c1_upper,S2mu,S2sd)
        
            I_FAR_area_rS1 = fncdf(t2c1_lower,S2mu,S2sd)
            C_HR_area_rS1  = fncdf(t2c1_lower,S1mu,S1sd)
        
            est_FAR2_rS2.append((I_FAR_area_rS2 + 1e-9) / (I_area_rS2 + 1e-9))
            est_HR2_rS2.append((C_HR_area_rS2 + 1e-9)  / (C_area_rS2 + 1e-9))
            
            est_FAR2_rS1.append((I_FAR_area_rS1 + 1e-9) / (I_area_rS1 + 1e-9))
            est_HR2_rS1.append((C_HR_area_rS1 + 1e-9) / (C_area_rS1 + 1e-9))
        
        # print("obs_FAR2_rS1 ", obs_FAR2_rS1)
        # print("obs_FAR2_rS2 ", obs_FAR2_rS2)
        # print("obs_HR2_rS1 ", obs_HR2_rS1)
        # print("obs_HR2_rS2 ", obs_HR2_rS2)
        # print("est_FAR2_rS1 ", est_FAR2_rS1)
        # print("est_FAR2_rS2 ", est_FAR2_rS2)
        # print("est_HR2_rS1 ", est_HR2_rS1)
        # print("est_HR2_rS2 ", est_HR2_rS2)

        t2c1x = [-np.inf]
        t2c1x.extend(t2c1[0:(nRatings-1)])
        t2c1x.append(t1c1)
        t2c1x.extend(t2c1[(nRatings-1):])
        t2c1x.append(np.inf)

        prC_rS1 = [( fncdf(t2c1x[i+1],S1mu,S1sd) - fncdf(t2c1x[i],S1mu,S1sd) + 1e-9 ) / (C_area_rS1 + 1e-9) for i in range(nRatings)]
        prI_rS1 = [( fncdf(t2c1x[i+1],S2mu,S2sd) - fncdf(t2c1x[i],S2mu,S2sd) + 1e-9) / (I_area_rS1 + 1e-9) for i in range(nRatings)]

        prC_rS2 = [( (1-fncdf(t2c1x[nRatings+i],S2mu,S2sd)) - (1-fncdf(t2c1x[nRatings+i+1],S2mu,S2sd)) + 1e-9) / (C_area_rS2 + 1e-9) for i in range(nRatings)]
        prI_rS2 = [( (1-fncdf(t2c1x[nRatings+i],S1mu,S1sd)) - (1-fncdf(t2c1x[nRatings+i+1],S1mu,S1sd)) + 1e-9) / (I_area_rS2 + 1e-9) for i in range(nRatings)]
        

        # idealization constraint 1
        S1_ratio = []
        S2_ratio = []
            
        for C, I in zip(prC_rS1, prI_rS1):
            if(I * I_area_rS1) == 0:
                print("div 0 s1")
            S1_ratio.append((C * C_area_rS1 + 1e-9) / (I * I_area_rS1 + 1e-9))

        for C, I in zip(prC_rS2, prI_rS2):
            if(I * I_area_rS2) == 0:
                print("div 0 S2")
            S2_ratio.append((C * C_area_rS2 + 1e-9) / (I * I_area_rS2 + 1e-9))

        diff = []
        for i in range(len(S1_ratio) - 1):
            diff.append(S1_ratio[i] - S1_ratio[i + 1])
        for i in range(len(S2_ratio) - 1):
            diff.append(S2_ratio[i + 1] - S2_ratio[i])
        loss_idealization = -np.log(sum(diff))
        # print("S1_ratio:", S1_ratio)
        # print("S2_ratio:", S2_ratio)
        # print("diff:", diff)
        # if all(x >= beta and x < 1e5 for x in diff) == False:
        #     return 1e+30

        # if meta_d1 >= d1:
        #     return 1e+300
        
        # idealization constraint 2        
        # print(criterion_distance)
        # alpha = 2
        # calculate logL
        # logL = np.sum([
        #         nC_rS1[i]*np.log(prC_rS1[i]) \
        #         + nI_rS1[i]*np.log(prI_rS1[i]) \
        #         + nC_rS2[i]*np.log(prC_rS2[i]) \
        #         + nI_rS2[i]*np.log(prI_rS2[i]) for i in range(nRatings)]) 
        # logL += alpha * np.log(criterion_distance)
        
        FAR2_rS1_error = sum([(a - b) ** 2 for a, b in zip(obs_FAR2_rS1, est_FAR2_rS1)]) / len(obs_FAR2_rS1)
        HR2_rS1_error = sum([(a - b) ** 2 for a, b in zip(obs_HR2_rS1, est_HR2_rS1)]) / len(obs_HR2_rS1)
        FAR2_rS2_error = sum([(a - b) ** 2 for a, b in zip(obs_FAR2_rS2, est_FAR2_rS2)]) / len(obs_FAR2_rS2)
        HR2_rS2_error = sum([(a - b) ** 2 for a, b in zip(obs_HR2_rS2, est_HR2_rS2)]) / len(obs_HR2_rS2)
        loss = FAR2_rS1_error + FAR2_rS2_error + HR2_rS1_error + HR2_rS2_error

        if np.isinf(loss) or np.isnan(loss):
    #        logL=-np.inf
            loss=1e+300 # returning "-inf" may cause optimize.minimize() to fail
        #self.loss_record.append(-logL)
        return loss
    
    def callback(self, x, inputObj):
        loss = self.__fit_meta_d_logL(x)
        self.loss_record.append(loss)


    def __idealization_cons_func(self, x, inputObj):
        nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv, beta = inputObj
        print("x:", x)
            
        meta_d1 = x[0]
        t2c1    = x[1:]
            
        S1mu = -meta_d1/2
        S1sd = 1
        S2mu = meta_d1/2
        S2sd = S1sd/s

        S1mu = S1mu - eval(constant_criterion)
        S2mu = S2mu - eval(constant_criterion)

        C_area_rS1 = fncdf(t1c1,S1mu,S1sd)
        I_area_rS1 = fncdf(t1c1,S2mu,S2sd)
        
        C_area_rS2 = 1-fncdf(t1c1,S2mu,S2sd)
        I_area_rS2 = 1-fncdf(t1c1,S1mu,S1sd)

        t1c1 = 0
        t2c1x = [-np.inf]
        t2c1x.extend(t2c1[0:(nRatings-1)])
        t2c1x.append(t1c1) #t1c1 is zero
        t2c1x.extend(t2c1[(nRatings-1):])
        t2c1x.append(np.inf)
        
        for i in range(len(t2c1x) - 1):
            if t2c1x[i + 1] <= t2c1x[i]:
                print("criterion error")
                return [-1.0] * ((nRatings - 1) * 2)
                # print("wrong return")
        
        prC_rS1 = [( fncdf(t2c1x[i+1],S1mu,S1sd) - fncdf(t2c1x[i],S1mu,S1sd) ) / C_area_rS1  for i in range(nRatings)]
        prI_rS1 = [( fncdf(t2c1x[i+1],S2mu,S2sd) - fncdf(t2c1x[i],S2mu,S2sd) ) / I_area_rS1 for i in range(nRatings)]

        prC_rS2 = [( (1-fncdf(t2c1x[nRatings+i],S2mu,S2sd)) - (1-fncdf(t2c1x[nRatings+i+1],S2mu,S2sd)) ) / C_area_rS2 for i in range(nRatings)]
        prI_rS2 = [( (1-fncdf(t2c1x[nRatings+i],S1mu,S1sd)) - (1-fncdf(t2c1x[nRatings+i+1],S1mu,S1sd)) ) / I_area_rS2 for i in range(nRatings)]
        print("prC_rS1:", prC_rS1)
        print("prI_rS1:", prI_rS1)
        print("prC_rS2:", prC_rS2)
        print("prI_rS2:", prI_rS2)
        # print("11111")
        for pr in [prC_rS1, prI_rS1, prC_rS2, prI_rS2]:
            if any(np.isnan(pr)) or any(np.isinf(pr)):
                print("fncdf calculation resulted in NaN or Inf")
                return [-1.0] * ((nRatings - 1) * 2)


        if (all(x > 0.0 for x in prC_rS1) and all(x > 0.0 for x in prI_rS1) and all(x > 0.0 for x in prC_rS2) and all(x > 0.0 for x in prI_rS2)) == False:
            print("area_list error ")
            return [-1.0] * ((nRatings - 1) * 2)

        
        S1_ratio = []
        S2_ratio = []
            

            
        for C, I in zip(prC_rS1, prI_rS1):
            if C <= 0 or I <= 0:
                print(C)
                print(I)
                print("negnum")
            S1_ratio.append((C / I) * (C_area_rS1 / I_area_rS1))

        for C, I in zip(prC_rS2, prI_rS2):
            if C <= 0 or I <= 0:
                print(C)
                print(I)
                print("negnum")
            S2_ratio.append((C / I) * (C_area_rS2 / I_area_rS2))
            
        # print(S1_ratio)
        # print(S2_ratio)


        diff = []
        for i in range(len(S1_ratio) - 1):
            diff.append(S1_ratio[i] - S1_ratio[i + 1])
        for i in range(len(S2_ratio) - 1):
            diff.append(S2_ratio[i + 1] - S2_ratio[i])
            # print(diff)
        diff = np.array(diff)
        for item in diff:
            if np.isinf(item) or np.isnan(item):
                print("nan or inf in diff")
                return [-1.0] * ((nRatings - 1) * 2)
        print(len(diff))
        print(diff)
        return diff

    def fit_meta_d_MLE(self, nR_S1, nR_S2, beta, p, s = 1, fncdf = norm.cdf, fninv = norm.ppf):
        
        print("beta:", beta)
        """
        function fit = fit_meta_d_MLE(nR_S1, nR_S2, s, fncdf, fninv)

        % fit = fit_meta_d_MLE(nR_S1, nR_S2, s, fncdf, fninv)
        %
        % Given data from an experiment where an observer discriminates between two
        % stimulus alternatives on every trial and provides confidence ratings,
        % provides a type 2 SDT analysis of the data.
        %
        % INPUTS
        %
        % * nR_S1, nR_S2
        % these are vectors containing the total number of responses in
        % each response category, conditional on presentation of S1 and S2.
        %
        % e.g. if nR_S1 = [100 50 20 10 5 1], then when stimulus S1 was
        % presented, the subject had the following response counts:
        % responded S1, rating=3 : 100 times
        % responded S1, rating=2 : 50 times
        % responded S1, rating=1 : 20 times
        % responded S2, rating=1 : 10 times
        % responded S2, rating=2 : 5 times
        % responded S2, rating=3 : 1 time
        %
        % The ordering of response / rating counts for S2 should be the same as it
        % is for S1. e.g. if nR_S2 = [3 7 8 12 27 89], then when stimulus S2 was
        % presented, the subject had the following response counts:
        % responded S1, rating=3 : 3 times
        % responded S1, rating=2 : 7 times
        % responded S1, rating=1 : 8 times
        % responded S2, rating=1 : 12 times
        % responded S2, rating=2 : 27 times
        % responded S2, rating=3 : 89 times
        %
        % N.B. if nR_S1 or nR_S2 contain zeros, this may interfere with estimation of
        % meta-d'.
        %
        % Some options for dealing with response cell counts containing zeros are:
        % 
        % (1) Add a small adjustment factor, e.g. adj_f = 1/(length(nR_S1), to each 
        % input vector:
        % 
        % adj_f = 1/length(nR_S1);
        % nR_S1_adj = nR_S1 + adj_f;
        % nR_S2_adj = nR_S2 + adj_f;
        % 
        % This is a generalization of the correction for similar estimation issues of
        % type 1 d' as recommended in
        % 
        % Hautus, M. J. (1995). Corrections for extreme proportions and their biasing 
        %     effects on estimated values of d'. Behavior Research Methods, Instruments, 
        %     & Computers, 27, 46-51.
        %     
        % When using this correction method, it is recommended to add the adjustment 
        % factor to ALL data for all subjects, even for those subjects whose data is 
        % not in need of such correction, in order to avoid biases in the analysis 
        % (cf Snodgrass & Corwin, 1988).
        % 
        % (2) Collapse across rating categories.
        % 
        % e.g. if your data set has 4 possible confidence ratings such that length(nR_S1)==8,
        % defining new input vectors
        % 
        % nR_S1_new = [sum(nR_S1(1:2)), sum(nR_S1(3:4)), sum(nR_S1(5:6)), sum(nR_S1(7:8))];
        % nR_S2_new = [sum(nR_S2(1:2)), sum(nR_S2(3:4)), sum(nR_S2(5:6)), sum(nR_S2(7:8))];
        % 
        % might be sufficient to eliminate zeros from the input without using an adjustment.
        %
        % * s
        % this is the ratio of standard deviations for type 1 distributions, i.e.
        %
        % s = sd(S1) / sd(S2)
        %
        % if not specified, s is set to a default value of 1.
        % For most purposes, we recommend setting s = 1. 
        % See http://www.columbia.edu/~bsm2105/type2sdt for further discussion.
        %
        % * fncdf
        % a function handle for the CDF of the type 1 distribution.
        % if not specified, fncdf defaults to @normcdf (i.e. CDF for normal
        % distribution)
        %
        % * fninv
        % a function handle for the inverse CDF of the type 1 distribution.
        % if not specified, fninv defaults to @norminv
        %
        % OUTPUT
        %
        % Output is packaged in the struct "fit." 
        % In the following, let S1 and S2 represent the distributions of evidence 
        % generated by stimulus classes S1 and S2.
        % Then the fields of "fit" are as follows:
        % 
        % fit.da        = mean(S2) - mean(S1), in room-mean-square(sd(S1),sd(S2)) units
        % fit.s         = sd(S1) / sd(S2)
        % fit.meta_da   = meta-d' in RMS units
        % fit.M_diff    = meta_da - da
        % fit.M_ratio   = meta_da / da
        % fit.meta_ca   = type 1 criterion for meta-d' fit, RMS units
        % fit.t2ca_rS1  = type 2 criteria of "S1" responses for meta-d' fit, RMS units
        % fit.t2ca_rS2  = type 2 criteria of "S2" responses for meta-d' fit, RMS units
        %
        % fit.S1units   = contains same parameters in sd(S1) units.
        %                 these may be of use since the data-fitting is conducted  
        %                 using parameters specified in sd(S1) units.
        % 
        % fit.logL          = log likelihood of the data fit
        %
        % fit.est_HR2_rS1  = estimated (from meta-d' fit) type 2 hit rates for S1 responses
        % fit.obs_HR2_rS1  = actual type 2 hit rates for S1 responses
        % fit.est_FAR2_rS1 = estimated type 2 false alarm rates for S1 responses
        % fit.obs_FAR2_rS1 = actual type 2 false alarm rates for S1 responses
        % 
        % fit.est_HR2_rS2  = estimated type 2 hit rates for S2 responses
        % fit.obs_HR2_rS2  = actual type 2 hit rates for S2 responses
        % fit.est_FAR2_rS2 = estimated type 2 false alarm rates for S2 responses
        % fit.obs_FAR2_rS2 = actual type 2 false alarm rates for S2 responses
        %
        % If there are N ratings, then there will be N-1 type 2 hit rates and false
        % alarm rates. 
        """
        
        # check inputs
        if (len(nR_S1) % 2)!=0: 
            raise('input arrays must have an even number of elements')
        if len(nR_S1)!=len(nR_S2):
            raise('input arrays must have the same number of elements')
        if any(np.array(nR_S1) == 0) or any(np.array(nR_S2) == 0):
            print(' ')
            print('WARNING!!')
            print('---------')
            print('Your inputs')
            print(' ')
            print('nR_S1:')
            print(nR_S1)
            print('nR_S2:')
            print(nR_S2)
            print(' ')
            print('contain zeros! This may interfere with proper estimation of meta-d''.')
            print('See ''help fit_meta_d_MLE'' for more information.')
            print(' ')
            print(' ')
        
        nRatings = int(len(nR_S1) / 2)  # number of ratings in the experiment
        nCriteria = int(2*nRatings - 1) # number criteria to be fitted
        

        
        """
        prepare other inputs for scipy.optimize.minimum()
        """
        # select constant criterion type
        constant_criterion = 'meta_d1 * (t1c1 / d1)' # relative criterion
        
        # set up initial guess at parameter values
        ratingHR  = []
        ratingFAR = []
        for c in range(1,int(nRatings*2)):
            ratingHR.append(sum(nR_S2[c:]) / sum(nR_S2))
            ratingFAR.append(sum(nR_S1[c:]) / sum(nR_S1))
        
        # obtain index in the criteria array to mark Type I and Type II criteria
        t1_index = nRatings-1
        t2_index = list(set(list(range(0,2*nRatings-1))) - set([t1_index]))
        
        d1 = (1/s) * (fninv( ratingHR[t1_index] ) - fninv( ratingFAR[t1_index] )) 
        meta_d1 = d1
        
        
        c1 = (-1/(1+s)) * ( fninv( ratingHR ) + fninv( ratingFAR ) )
        # print(c1)
        t1c1 = c1[t1_index]
        t2c1 = c1[t2_index]
        
        # initial values for the minimization function
        
        random.seed(beta)
        guess = [random.uniform(0.0, d1)]
        guess.extend(list(t2c1 - eval(constant_criterion)))
        # print(guess)

        """
        set up constraints for scipy.optimize.minimum()
        """
        # parameters
        # meta-d' - 1
        # t2c     - nCriteria-1
        # constrain type 2 criteria values,
        # such that t2c(i) is always <= t2c(i+1)
        # want t2c(i)   <= t2c(i+1) 
        # -->  t2c(i+1) >= t2c(i) + 1e-5 (i.e. very small deviation from equality) 
        # -->  t2c(i) - t2c(i+1) <= -1e-5
        
        A = []
        ub = []
        lb = []
        for ii in range(nCriteria-2):
            tempArow = []
            if ii == (nCriteria - 2) // 2:
                tempArow.extend(np.zeros(ii+1))
                tempArow.append(1)
                tempArow.extend(np.zeros((nCriteria-2)-ii))
                A.append(tempArow)
                ub.append(-0.05)
                lb.append(-np.inf)

                tempArow = []
                tempArow.extend(np.zeros(ii+2))
                tempArow.append(-1)
                tempArow.extend(np.zeros((nCriteria-2)-ii-1))
                A.append(tempArow)
                ub.append(-0.05)
                lb.append(-np.inf)
            else:
                tempArow.extend(np.zeros(ii+1))
                tempArow.extend([1, -1])
                tempArow.extend(np.zeros((nCriteria-2)-ii-1))
                A.append(tempArow)
                ub.append(-0.05)
                lb.append(-np.inf)
            
        # lower bounds on parameters
        LB = []
        LB.append(0)                              # meta-d'
        LB.extend(-20*np.ones((nCriteria-1)//2))    # criteria lower than t1c
        LB.extend(np.zeros((nCriteria-1)//2))       # criteria higher than t1c
        
        # upper bounds on parameters
        UB = []
        UB.append(d1)                           # meta-d'
        UB.extend(np.zeros((nCriteria-1)//2))      # criteria lower than t1c
        UB.extend(20*np.ones((nCriteria-1)//2))    # criteria higher than t1c
        
        
        # other inputs for the minimization function
        inputObj = [nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv, beta, p]        
        bounds = Bounds(LB,UB)
        linear_constraint = LinearConstraint(A,lb,ub)
        
        

        
        idealization_cons_func_fixed = partial(self.__idealization_cons_func, inputObj=inputObj)
        nonlinear_constraint = NonlinearConstraint(idealization_cons_func_fixed, [beta] * ((nRatings - 1) * 2), [np.inf] * ((nRatings - 1) * 2))
                

        # minimization of negative log-likelihood
        # optimize method: "trust-constr", "L-BFGS-B", "SLsQP"
        # results = minimize(self.__fit_meta_d_logL, guess, args = (inputObj), method='trust-constr',
        #                 jac='2-point', hess=SR1(),
        #                 constraints = [linear_constraint],
        #                 options = {'verbose': 0, "maxiter": 1000}, bounds = bounds,
        #                 )
        
        results = minimize(self.__fit_meta_d_logL, guess, args = (inputObj), method='trust-constr',
                        jac='2-point', hess=SR1(), 
                        constraints=[linear_constraint, nonlinear_constraint],
                        options = {'verbose': 0, "maxiter": 10000000}, bounds = bounds,)
        
        # plt.plot(self.loss_record, marker='-')
        # plt.title('Loss vs Iterations')
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.grid(True)
        # plt.show()


        # quickly process some of the output
        meta_d1 = results.x[0] 
        t2c1    = results.x[1:] + eval(constant_criterion)
        # t2c1    = results.x[1:]
        logL    = results.fun
        is_success = results.success
        if is_success:
            print("successful!")
        else:
            print("failed!")
        
        # data is fit, now to package it...
        # find observed t2FAR and t2HR 
        
        # I_nR and C_nR are rating trial counts for incorrect and correct trials
        # element i corresponds to # (in)correct w/ rating i
        I_nR_rS2 = nR_S1[nRatings:]
        I_nR_rS1 = list(np.flip(nR_S2[0:nRatings],axis=0))
        
        C_nR_rS2 = nR_S2[nRatings:];
        C_nR_rS1 = list(np.flip(nR_S1[0:nRatings],axis=0))
        
        obs_FAR2_rS2 = [sum( I_nR_rS2[(i+1):] ) / sum(I_nR_rS2) for i in range(nRatings-1)]
        obs_HR2_rS2 = [sum( C_nR_rS2[(i+1):] ) / sum(C_nR_rS2) for i in range(nRatings-1)]
        obs_FAR2_rS1 = [sum( I_nR_rS1[(i+1):] ) / sum(I_nR_rS1) for i in range(nRatings-1)]
        obs_HR2_rS1 = [sum( C_nR_rS1[(i+1):] ) / sum(C_nR_rS1) for i in range(nRatings-1)]
        
        # find estimated t2FAR and t2HR
        S1mu = -meta_d1/2
        S1sd = 1
        S2mu =  meta_d1/2
        S2sd = S1sd/s;
        
        mt1c1 = eval(constant_criterion)
        
        # #绘制正态分布图像
        # x = np.linspace(-2, 2, 10000)
        # pdf1 = norm.pdf(x, S1mu, S1sd)
        # pdf2 = norm.pdf(x, S2mu, S2sd)

        # plt.figure(figsize=(8, 6))
        # plt.plot(x, pdf1, label="S1")
        # plt.plot(x, pdf2, label="S2")

        # markers4S1 = ["v", "s", "D"]
        # markers4S2 = ["D", "s", "v"]

        # for i in range((nRatings - 1)):
        #     plt.scatter(t2c1[i], 0, color="red", marker=markers4S1[i], label=f"C_S1_{nRatings - 1 - i}")
              
        # for i in range((nRatings - 1), len(t2c1)):
        #     plt.scatter(t2c1[i], 0, color="blue", marker=markers4S2[i - nRatings + 1], label=f"C_S2_{i - nRatings + 2}")
            

        # plt.scatter(mt1c1, 0, color="green", marker='o', label="C1")
        


        # plt.axhline(0, color='black', linewidth=0.5)  # 绘制横坐标轴
        # plt.axvline(0, color='black', linewidth=0.5)  # 绘制纵坐标轴
        # ax = plt.gca()  # 获取当前的轴
        # ax.spines['left'].set_position('zero')  # 纵坐标轴的位置
        # ax.spines['bottom'].set_position('zero')  # 横坐标轴的位置
        # ax.spines['right'].set_color('none')  # 隐藏右边框
        # ax.spines['top'].set_color('none')  # 隐藏上边框
        # plt.legend()
        # plt.show()

        C_area_rS2 = 1-fncdf(mt1c1,S2mu,S2sd)
        I_area_rS2 = 1-fncdf(mt1c1,S1mu,S1sd)
        
        C_area_rS1 = fncdf(mt1c1,S1mu,S1sd)
        I_area_rS1 = fncdf(mt1c1,S2mu,S2sd)
        
        est_FAR2_rS2 = []
        est_HR2_rS2 = []
        
        est_FAR2_rS1 = []
        est_HR2_rS1 = []
        
        
        for i in range(nRatings-1):
            
            t2c1_lower = t2c1[(nRatings-1)-(i+1)]
            t2c1_upper = t2c1[(nRatings-1)+i]
                
            I_FAR_area_rS2 = 1-fncdf(t2c1_upper,S1mu,S1sd)
            C_HR_area_rS2  = 1-fncdf(t2c1_upper,S2mu,S2sd)
        
            I_FAR_area_rS1 = fncdf(t2c1_lower,S2mu,S2sd)
            C_HR_area_rS1  = fncdf(t2c1_lower,S1mu,S1sd)
        
            est_FAR2_rS2.append(I_FAR_area_rS2 / I_area_rS2)
            est_HR2_rS2.append(C_HR_area_rS2 / C_area_rS2)
            
            est_FAR2_rS1.append(I_FAR_area_rS1 / I_area_rS1)
            est_HR2_rS1.append(C_HR_area_rS1 / C_area_rS1)
        
        
        # package output
        fit = {}
        fit['da']       = np.sqrt(2/(1+s**2)) * s * d1
        
        fit['s']        = s
        
        fit['meta_da']  = np.sqrt(2/(1+s**2)) * s * meta_d1
        
        fit['M_diff']   = fit['meta_da'] - fit['da']
        
        fit['M_ratio']  = fit['meta_da'] / fit['da']
        
        mt1c1         = eval(constant_criterion)
        fit['meta_ca']  = ( np.sqrt(2)*s / np.sqrt(1+s**2) ) * mt1c1
        
        t2ca          = ( np.sqrt(2)*s / np.sqrt(1+s**2) ) * np.array(t2c1)
        fit['t2ca_rS1']     = t2ca[0:nRatings-1]
        fit['t2ca_rS2']     = t2ca[(nRatings-1):]
        
        fit['S1units'] = {}
        fit['S1units']['d1']        = d1
        fit['S1units']['meta_d1']   = meta_d1 
        fit['S1units']['s']         = s
        fit['S1units']['meta_c1']   = mt1c1
        fit['S1units']['t2c1_rS1']  = t2c1[0:nRatings-1]
        fit['S1units']['t2c1_rS2']  = t2c1[(nRatings-1):]
        
        fit['logL']    = logL
        
        fit['est_HR2_rS1']  = est_HR2_rS1
        fit['obs_HR2_rS1']  = obs_HR2_rS1
        
        fit['est_FAR2_rS1'] = est_FAR2_rS1
        fit['obs_FAR2_rS1'] = obs_FAR2_rS1
        
        fit['est_HR2_rS2']  = est_HR2_rS2
        fit['obs_HR2_rS2']  = obs_HR2_rS2
        
        fit['est_FAR2_rS2'] = est_FAR2_rS2
        fit['obs_FAR2_rS2'] = obs_FAR2_rS2

        return fit
    
