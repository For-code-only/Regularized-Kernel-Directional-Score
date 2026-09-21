# RKDS equations and exact permutation computation from the v1 implementation.
SCREEN_B <- 999L
SCREEN_SIGMA <- 0.4
SCREEN_ALPHA <- 0.05
NUM_TOL <- 1e-12
NEGATIVE_ENERGY_TOL <- 1e-10

center_gram <- function(K) {
  m <- rowMeans(K)
  sweep(sweep(K,1L,m,"-"),2L,m,"-") + mean(m)
}
gaussian_gram <- function(x, sigma) {
  stopifnot(is.finite(sigma), sigma > 0)
  exp(-outer(x,x,"-")^2/(2*sigma^2))
}
standardize <- function(x) {
  if (length(x)<3L || any(!is.finite(x)) || !is.finite(sd(x)) || sd(x)==0)
    stop("Invalid or constant input")
  as.numeric((x-mean(x))/sd(x))
}
residual_operator <- function(Kc,theta) {
  n <- nrow(Kc)
  A <- Kc/n
  diag(A) <- diag(A)+theta
  theta*chol2inv(chol(A))
}
residual_energies <- function(R,Kc) {
  d <- rowSums((R %*% Kc)*R)
  if (any(!is.finite(d)) || min(d)< -NEGATIVE_ENERGY_TOL)
    stop("Residual energy is nonfinite or materially negative")
  list(d=pmax(d,0), truncated=sum(d<0), minimum=min(d))
}
normalized_score <- function(Kc,d,sigma3) {
  Lc <- center_gram(gaussian_gram(d,sigma3))
  n <- nrow(Kc)
  a <- sum(Kc*Kc)/n^2
  b <- sum(Lc*Lc)/n^2
  den <- sqrt(a*b)
  if (!is.finite(den) || den<=0)
    return(c(q=NA_real_,hsic=NA_real_,denominator=den,
             input_self_hsic=a,energy_self_hsic=b))
  numerator <- sum(Kc*Lc)/n^2
  q <- numerator/den
  if (!is.finite(q) || q < -1e-10 || q > 1+1e-10)
    stop("Normalized score outside its numerical range")
  c(q=min(1,max(0,q)),hsic=numerator,denominator=den,
    input_self_hsic=a,energy_self_hsic=b)
}

.accel <- new.env(parent=globalenv())
.accel$fun <- NULL
screen_hsic <- function(x,y,seed,B=SCREEN_B,permutations=NULL) {
  K <- center_gram(gaussian_gram(x,SCREEN_SIGMA))
  L <- center_gram(gaussian_gram(y,SCREEN_SIGMA))
  n <- length(x)
  if (is.null(permutations)) {
    set.seed(seed)
    permutations <- replicate(B,sample.int(n))
  }
  B <- ncol(permutations)
  obs <- sum(K*L)
  perm <- if (is.function(.accel$fun)) .accel$fun(K,L,permutations) else
    vapply(seq_len(B),function(b) {
      ix <- permutations[,b]
      sum(K*L[ix,ix,drop=FALSE])
    },numeric(1L))
  # Identity and other numerical ties count as at least the observed value.
  tolerance <- 64*.Machine$double.eps*max(1,abs(obs))
  p <- (1+sum(perm >= obs-tolerance))/(B+1)
  list(p=p,hsic=obs/n^2,rejected=p<=SCREEN_ALPHA,B=B)
}
