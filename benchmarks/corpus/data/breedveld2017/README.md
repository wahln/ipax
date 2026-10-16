# Breedveld et al. (2017) TROTS reference timings

`trots_reference.json` holds the per-problem iteration counts, run times,
problem sizes and module timings that Breedveld et al. report for the 120
TROTS problems of their interior-point solver study. The ipax TROTS runner
(`benchmarks/runners/trots.py`) shows them next to its own results; they are
read by `benchmarks/corpus/trots_reference.py`.

## Attribution

> Breedveld, S., van den Berg, B. & Heijmen, B. (2017). "An interior-point
> implementation developed and tuned for radiation therapy treatment
> planning." *Computational Optimization and Applications* 68(2), 209–242.
> <https://doi.org/10.1007/s10589-017-9919-4>
>
> Electronic supplementary material 2 (`10589_2017_9919_MOESM2_ESM.xlsx`),
> © 2017 The Author(s), licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Converted to JSON
> without modification of the values by the ipax project, 2026-09-27.

The same workbook ships with the TROTS dataset as
`Results/Supplementary_Table.xlsx`. The conversion
(`benchmarks/tools/convert_breedveld2017_esm.py`) keeps every stored cell value
as-is. It leaves out the formula-derived columns (speed-up factors, time per
iteration, column sums) and adds each problem's TROTS case name and tumour
site. The script's docstring decodes every spreadsheet column.

## License

This directory is licensed under **CC BY 4.0** (see [`LICENSE`](./LICENSE)).
It is the exception: the rest of ipax is licensed under Apache-2.0 (see the
repository [`LICENSE`](../../../../LICENSE)). The data is not part of the
`ipax` wheel. It ships only with the source distribution, next to this
notice.
