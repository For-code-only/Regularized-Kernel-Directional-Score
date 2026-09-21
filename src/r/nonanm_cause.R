# Population expectations for the NEW cause library. IDs follow the original
# NOISE library (not synthetic_cause_names). All four causes have E[X]=0, Var=1.
# This extension does not alter synthetic_final.R or any inference method.

nonanm_cause_ids <- c(1L, 3L, 5L, 9L)
nonanm_cause_names <- c('Gaussian', 'Laplace', 'Gamma_shape2_scale1', 'bimodal')

nonanm_expectation <- function(h, cause_id, rel.tol=1e-10, abs.tol=1e-11) {
  if(length(cause_id)!=1L || is.na(cause_id) || !(cause_id %in% nonanm_cause_ids))
    stop('Unknown new non-ANM cause distribution')
  quad <- function(fun, lo, hi) {
    value <- integrate(fun,lo,hi,subdivisions=2000L,rel.tol=rel.tol,
                       abs.tol=abs.tol,stop.on.error=TRUE)$value
    if(!is.finite(value)) stop('Nonfinite population expectation')
    value
  }
  # Only the numerical integration of the standard-normal coordinate is
  # bounded at 12 SD. Generated Gaussian observations are NEVER truncated.
  # Selected mechanisms are bounded or at most cubic, so the missing normal
  # tail contribution through squared signal is negligible at this tolerance.
  switch(as.character(as.integer(cause_id)),
    `1` = quad(function(z) h(z)*dnorm(z),-12,12),
    `3` = quad(function(z) .5*(h(z/sqrt(2))+h(-z/sqrt(2)))*exp(-z),0,Inf),
    `5` = quad(function(z) h((z-2)/sqrt(2))*dgamma(z,shape=2,scale=1),0,Inf),
    `9` = .5*quad(function(z) h((2+.5*z)/sqrt(4.25))*dnorm(z),-12,12) +
          .5*quad(function(z) h((-2+.5*z)/sqrt(4.25))*dnorm(z),-12,12))
}

nonanm_signal_moments <- function(function_id,cause_id,a,b) {
  if(length(function_id)!=1L || is.na(function_id) ||
     !(function_id %in% c(7L,8L,6L,4L,10L,11L)) ||
     length(a)!=1L || !is.finite(a) || a<.8 || a>1.2 ||
     length(b)!=1L || !is.finite(b) || b < -.2 || b>.2)
    stop('New non-ANM population-moment arguments outside frozen bounds')
  g <- function(x) synthetic_mechanism(a*x+b,function_id)
  mu <- nonanm_expectation(g,cause_id)
  variance <- nonanm_expectation(function(x) (g(x)-mu)^2,cause_id)
  if(!is.finite(variance) || variance<=0) stop('Nonpositive signal population variance')
  list(mean=mu,sd=sqrt(variance))
}
