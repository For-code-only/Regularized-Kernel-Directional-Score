# Pre-specified KCDC settings. No baseline grid selection is performed.
kcdc_primary_configuration <- function() {
  list(lambda=1e-6,kernel_in="log",kernel_out="rq",h_in=1,h_out=1)
}
