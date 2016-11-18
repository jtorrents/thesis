library(survival)
library(stargazer)

df <- read.csv('../data/survival_python_df.csv')

# Stratifyed by tenure and interactions of contributions and core with time for proportional hazards
# http://cran.r-project.org/doc/contrib/Fox-Companion/appendix-cox-regression.pdf

fit.cox <- coxph(Surv(rstart, rstop, status) ~ strata(tenure) + total_accepted_peps + contributions + colaborators + dcentrality + closeness, data=df)
fit.cox.knum <- coxph(Surv(rstart, rstop, status) ~ strata(tenure) + total_accepted_peps + contributions + colaborators + dcentrality + closeness + knum, data=df)
fit.cox.top <- coxph(Surv(rstart, rstop, status) ~ strata(tenure) + total_accepted_peps + contributions + colaborators + dcentrality + closeness + top, data=df)
fit.cox.all <- coxph(Surv(rstart, rstop, status) ~ strata(tenure) + total_accepted_peps + contributions + colaborators + dcentrality + closeness + knum + top, data=df)

summary(fit.cox.all)

stargazer(fit.cox, fit.cox.knum, fit.cox.top, fit.cox.all,
            out = '../tables/table_survival.tex',
            #se = NULL, # list(rob.std.err.1, rob.std.err.2, rob.std.err.3, rob.std.err.4)
            title="Survival Analysis: Cox proportional hazards regression model",
            align=TRUE,
            dep.var.labels=c("Time active in the project"),
            covariate.labels=c(
                "Total accepted PEPs",
                "Contributions",
                "Collaborators",
                "Degree Centrality",
                "Closeness",
                "k-component number",
                "Top connectivity level"
            ),
            star.cutoffs = c(0.05, 0.01, 0.001),
            keep.stat = c("n", "ll", "rsq", "max.rsq"),
            no.space=TRUE)


aafit <- aareg(Surv(rstart, rstop, status) ~ tenure + log(contributions) + colaborators + degree + knum +cluster(id), data=df)

# Checking model assumptions

# Checking Proportional Hazards
zph <- cox.zph(fit.cox.all)
par(mfrow=c(2,3))
plot(zph)

# Influential Observations
dfbeta <- residuals(fit.cox.all, type='dfbeta')
par(mfrow=c(2,3))
for (j in 1:5) {
     plot(dfbeta[,j], ylab=names(coef(fit.cox.all))[j])
     abline(h=0, lty=2)
}
# looks good

# Nonlinearity
par(mfrow=c(2,2))
res <- residuals(fit.cox.all, type='martingale')
X <- as.matrix(df[,c("top", "knum")]) # matrix of covariates
par(mfrow=c(2,2))
for (j in 1:2) { # residual plots
     plot(X[,j], res, xlab=c("top", "knum")[j], ylab="residuals")
abline(h=0, lty=2)
lines(lowess(X[,j], res, iter=0))
}

b <- coef(fit.cox.all)[c(1,4)] # regression coefficients
for (j in 1:2) { # partial-residual plots
     plot(X[,j], b[j]*X[,j] + res, xlab=c("top", "degree")[j],
         ylab="component+residual")
     abline(lm(b[j]*X[,j] + res ~ X[,j]), lty=2)
     lines(lowess(X[,j], b[j]*X[,j] + res, iter=0))
}

##
## For testing differences in groups
##
survdiff(Surv(df$period,df$status)~factor(df$knum))

survdiff(Surv(df$period,df$status)~factor(df$core))


