# VMOD temporal inputs

The `profiles33` and `profiles69` directories contain the frozen 15-minute
development arrays derived from the 366-day Open Power System Data window
beginning 1 January 2017. The
`profiles33_external2018` and `profiles69_external2018` directories contain the
non-overlapping external-selection and confirmation arrays derived from the
non-overlapping window beginning 2 January 2018. External values use normalization references frozen
from the development window; the metadata records that no external value was
clipped. Column counts differ because the common normalized temporal factors
are mapped to the fixed spatial allocations of the two benchmark feeders.

The profiles are public system-level temporal shapes, not measurements from an
IEEE benchmark feeder. The original source, field names, time window,
normalization, array dimensions and SHA-256 hashes are recorded in each
directory's `metadata.json`.
