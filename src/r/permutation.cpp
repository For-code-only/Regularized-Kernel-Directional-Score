#include <Rcpp.h>
// [[Rcpp::export]]
Rcpp::NumericVector extend1_permutation_sums(
    Rcpp::NumericMatrix K, Rcpp::NumericMatrix L, Rcpp::IntegerMatrix P) {
  const int n=K.nrow(), B=P.ncol();
  Rcpp::NumericVector out(B);
  for(int b=0; b<B; ++b) {
    Rcpp::checkUserInterrupt();
    double total=0.0;
    for(int j=0; j<n; ++j) {
      const int pj=P(j,b)-1;
      for(int i=0; i<n; ++i) total+=K(i,j)*L(P(i,b)-1,pj);
    }
    out[b]=total;
  }
  return out;
}
