# Negative binomial regression for Debian uploads
library(MASS)
library(stargazer)
library(pscl)

# Load dataframe
df <- read.csv('../data/debian_Wheezy_developers_df.csv')

#df['top'] <- ifelse(df['knum'] > 2, 1, 0)
##
## Control settings for GLM
##
ctrl <- glm.control(maxit=100)

##
## Negative binomial regression
##

model1 <- glm.nb(contributions ~ psizes + bugs + deps + tenure, data=df, control=ctrl)

model2 <- glm.nb(contributions ~ psizes + bugs + deps + tenure + degree_cent + closeness, data=df, control=ctrl)

model3 <- glm.nb(contributions ~ psizes + bugs + deps + tenure + degree_cent + closeness + clus_sq, data=df, control=ctrl)

model4 <- glm.nb(contributions ~ psizes + bugs + deps + tenure + degree_cent + closeness + clus_sq + knum, data=df, control=ctrl)

# Cheicking model assumptions
pois4 <- glm(contributions ~ psizes + bugs + deps + tenure + degree + closeness + clus_sq + knum, family="poisson", data=df, control=ctrl)

pchisq(2 * (logLik(model4) - logLik(pois4)), df=1, lower.tail=FALSE)
## 'log Lik.' 0 (df=10)
# This strongly suggests the negative binomial model, estimating the dispersion parameter,
# is more appropriate than the Poisson model.
(est <- cbind(Estimate = coef(model4), confint(model4)))
exp(est)


##
## Make a nice table
##
stargazer(model1, model2, model3, model4,
            title="Negative binomial model for Debian uploads", 
            dep.var.labels=c("Number of uploads by developer"), 
            type='latex',
            out = '../tables/table_debian_uploads_negative_binomial.tex',
            align=TRUE, 
            covariate.labels=c(
                "Intercept",
                "log(Package size)",
                "\\# of bugs reported",
                "\\# of package dependencies",
                "Developer tenure (years)",
                "Degree centrality",
                "Closeness",
                "Square clustering",
                "k-component number"
            ),
            #zero.component=FALSE,
            intercept.bottom = FALSE,
            no.space=TRUE,
            omit.stat=c("theta"),
            star.cutoffs = c(0.05, 0.01, 0.001))

