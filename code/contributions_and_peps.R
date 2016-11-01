library(plm)
library(lmtest)
library(stargazer)
# Plain and zero inflated negative binomial regression.
library(MASS)
library(pscl)


# Load data frame
contrib <- read.csv('../data/developer_contributions_df.csv')
# As a robustness check we can remove some years
contrib_2000 <- contrib[contrib['year'] > 1999,]
# And remove Guido van Rossum
#contrib_no_guido <- contrib[contrib['dev'] != 'Guido van Rossum',]

##
## Panel linear regression for contributions measured as lines of source code
##
# Test several models againt each other
# Regular OLS (for comparison)
fit.lm <- lm(log(contributions_sc) ~ degree_cent + collaborators + tenure + betweenness + clus_sq + as.factor(top) + knum, data=contrib)
summary(fit.lm)

# Pooling OLS
fit.po <- plm(log(contributions_sc) ~ degree_cent + collaborators + tenure + betweenness + clus_sq + as.factor(top) + knum, data=contrib, index=c("id", "year"), model="pooling")#, effect="time")

summary(fit.po)

# Fixed effects
fit.fe <- plm(log(contributions_sc) ~ degree_cent + collaborators + tenure + betweenness + clus_sq + as.factor(top) + knum, data=contrib, index=c("id", "year"), model="within")#, effect="time")

summary(fit.fe)

# Random effects
fit.re <- plm(log(contributions_sc) ~ degree_cent + collaborators + tenure + betweenness + clus_sq + as.factor(top) + knum, data=contrib, index=c("id", "year"), model="random")#, effect="time")

summary(fit.re)

# Hausman Test for Panel Models
# The Hausman test is based on the difference of the vectors of
# coefficients of two different models.
# Test wheter fixed effects is a better model than random effects
phtest(fit.fe, fit.po)
phtest(fit.re, fit.po)
phtest(fit.fe, fit.re)
# if p < 0.05, as with this case, then fixed effects is better

# Controlling for heteroskedasticity
# Robust covariance matrix estimation (Sandwich estimator)
coeftest(fit.fe, vcovHC(fit.fe, type = "HC4"))
# The vcovHC() function returns the variance-covariance matrix under the assumption of "HC" (Heteroskedasticity-consistent) estimation. The square roots of the diagonal of this matrix are the robust standard errors.
cov.fit.fe <- vcovHC(fit.fe, type = "HC4")
rob.std.err <- sqrt(diag(cov.fit.fe))
# HC4 is a modern method for small samples with high influential observations
# Cribari & Neto (2004). "Asymptotic Inference Under Heteroskedasticity of Unknown Form." 
# Computational Statistics & Data Analysis, 45, 215–233.

#
# Make a pretty table with the results
#
fit.fe.1 <- plm(log(contributions_sc) ~ degree_cent + collaborators + tenure + betweenness, data=contrib, index=c("id", "year"), model="within")
cov.fit.fe.1 <- vcovHC(fit.fe.1, type = "HC4")
rob.std.err.1 <- sqrt(diag(cov.fit.fe.1))

fit.fe.2 <- plm(log(contributions_sc) ~ degree_cent + collaborators + tenure + betweenness + clus_sq, data=contrib, index=c("id", "year"), model="within")
cov.fit.fe.2 <- vcovHC(fit.fe.2, type = "HC4")
rob.std.err.2 <- sqrt(diag(cov.fit.fe.2))

fit.fe.3 <- plm(log(contributions_sc) ~ degree_cent + collaborators + tenure + betweenness + clus_sq + as.factor(top), data=contrib, index=c("id", "year"), model="within")
cov.fit.fe.3 <- vcovHC(fit.fe.3, type = "HC4")
rob.std.err.3 <- sqrt(diag(cov.fit.fe.3))

fit.fe.4 <- plm(log(contributions_sc) ~ degree_cent + collaborators + tenure + betweenness + clus_sq + as.factor(top) + knum, data=contrib, index=c("id", "year"), model="within")
cov.fit.fe.4 <- vcovHC(fit.fe.4, type = "HC4")
rob.std.err.4 <- sqrt(diag(cov.fit.fe.4))

stargazer(fit.fe.1, fit.fe.2, fit.fe.3, fit.fe.4,
            title="Contributions Panel Regression Results", 
            dep.var.labels=c("Lines of Source Code"), 
            type='latex',
            out = '../tables/table_plm_contributions.tex',
            align=TRUE, 
            covariate.labels=c("Degree Centrality",
                                "Collaborators", 
                                "Tenure (years)", 
                                "Betweenness", 
                                "Square clustering", 
                                "Top connectivity level", 
                                "$\\kappa$-component number"
            ),
            no.space=TRUE,
            omit.stat=c("f", "rsq"),
            se = list(rob.std.err.1, rob.std.err.2, rob.std.err.3, rob.std.err.4),
            star.cutoffs = c(0.05, 0.01, 0.001))

##
## Zero inflated negative binomial models for number of accepted PEPs authored
##
# PEP mechanism started at 2000, so we use only data from 200 to 2014
contrib_2000 <- contrib[contrib['year'] > 1999,]
# http://www.ats.ucla.edu/stat/r/dae/nbreg.htm
# Regression Models for Count Data in R
# www.jstatsoft.org/v27/i08/paper

plain.negbin <- glm.nb(total_accepted_peps ~ factor(year) + log(contributions_sc) + degree_cent + tenure + collaborators + betweenness + clus_sq + factor(top) + knum, data=contrib_2000)

summary(plain.negbin)

# Test if top is statistically significant (previous model only used top=1 because it is a factor)
plain.negbin.ntop <- glm.nb(total_accepted_peps ~ factor(year) + log(contributions_sc) + degree_cent + tenure + collaborators + betweenness + clus_sq + knum, data=contrib_2000)
anova(plain.negbin, plain.negbin.ntop)
# Looks good

# Cheicking model assumptions
poisson.glm <- glm(total_accepted_peps ~ factor(year) + log(contributions_sc) + degree_cent + tenure + collaborators + betweenness + clus_sq + factor(top) + knum, family="poisson", data=contrib_2000)

pchisq(2 * (logLik(plain.negbin) - logLik(poisson.glm)), df=1, lower.tail=FALSE)
## 'log Lik.' 8.822025e-115 (df=24)
# This strongly suggests the negative binomial model, estimating the dispersion parameter,
# is more appropriate than the Poisson model.
(est <- cbind(Estimate = coef(plain.negbin), confint(plain.negbin)))
exp(est)

# Hurdle and Zero inflated versions of the negative binomial model
negbin.hurdle <- hurdle(total_accepted_peps ~ factor(year) + log(contributions_sc) + degree_cent + tenure + collaborators + betweenness + clus_sq + factor(top) + knum, dist="negbin", data=contrib_2000)

summary(negbin.hurdle)

negbin.zi <- zeroinfl(total_accepted_peps ~ factor(year) + log(contributions_sc) + degree_cent + tenure + collaborators + betweenness + clus_sq + factor(top) + knum, dist="negbin", data=contrib_2000, EM=TRUE)

summary(negbin.zi)

# Compare models
fm <- list("poisson"=poisson.glm, "plain.negbin"=plain.negbin, "hurdle"=negbin.hurdle, "zinfl"=negbin.zi)
# Examine the coefficients
sapply(fm, function(x) coef(x)[16:23])
# the standard errors
sapply(fm, function(x) sqrt(diag(vcov(x)))[16:23])
# the log likelihood
rbind(logLik = sapply(fm, function(x) round(logLik(x), digits = 0)), 
      Df = sapply(fm, function(x) attr(logLik(x), "df")))

# Compare actual and expected zero counts
# Note that by construction, the expected number of zero counts in the
# hurdle model matches the observed number. The zero inflated does pretty well.
round(c("Obs" = sum(contrib_2000$total_accepted_peps < 1),
        "poisson" = sum(dpois(0, fitted(poisson.glm))),
        "plain.negbin" = sum(dnbinom(0, mu = fitted(plain.negbin), size = plain.negbin$theta)),
        "hurdle" = sum(predict(negbin.hurdle, type = "prob")[,1]),
        "zinfl" = sum(predict(negbin.zi, type = "prob")[,1])))


# The most parsimounious models would be
fit <- zeroinfl(total_accepted_peps ~ tenure + factor(top), dist="negbin", data=contrib_2000, EM=TRUE)
# or
fit <- hurdle(total_accepted_peps ~ tenure + factor(top), dist = "negbin", data=contrib_2000)
# but the point here is the theoretical argument


# Make a nice table
zinfl.1 <- zeroinfl(total_accepted_peps ~ factor(year) + log(contributions_sc) + degree_cent + tenure + collaborators + betweenness + clus_sq, dist="negbin", data=contrib_2000, EM=TRUE)

zinfl.2 <- zeroinfl(total_accepted_peps ~ factor(year) + log(contributions_sc) + degree_cent + tenure + collaborators + betweenness + clus_sq + factor(top), dist="negbin", data=contrib_2000, EM=TRUE)

zinfl.3 <- zeroinfl(total_accepted_peps ~ factor(year) + log(contributions_sc) + degree_cent + tenure + collaborators + betweenness + clus_sq + knum, dist="negbin", data=contrib_2000, EM=TRUE)

zinfl.4 <- zeroinfl(total_accepted_peps ~ factor(year) + log(contributions_sc) + degree_cent + tenure + collaborators + betweenness + clus_sq + factor(top) + knum, dist="negbin", data=contrib_2000, EM=TRUE)


stargazer(zinfl.1, zinfl.2, zinfl.3, zinfl.4,
            title="Zero Inflated negative binomial model for PEPs", 
            dep.var.labels=c("Total number of accepted PEPs authored"), 
            type='latex',
            out = '../tables/table_zinfl_peps.tex',
            align=TRUE, 
            covariate.labels=c(
                "log(\\# of lines of code authored)",
                "Degree Centrality",
                "Tenure (years)", 
                "Collaborators", 
                "Betweenness", 
                "Square clustering", 
                "Top connectivity level", 
                "$\\kappa$-component number"
            ),
            zero.component=FALSE,
            omit="year",
            omit.labels="Year dummies: ",
            no.space=TRUE,
            #omit.stat=c("f", "rsq"),
            star.cutoffs = c(0.05, 0.01, 0.001))








