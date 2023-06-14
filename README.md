# Explainable Data Decompositions with Desc

This repository provides a Julia library that implements the Desc algorithm. By leveraging the Bayesian information criterion and maximum entropy modeling, Desc efficiently discovers concise sets of informative higher-order feature interactions (i.e., patterns). Desc highlights commonalities and differences between groups by associating patterns with sets of groups if the pattern is characteristic for the groups.

The code is a from-scratch implementation of algorithms described in the [paper](https://doi.org/10.1609/aaai.v34i04.5780).
 
```
Dalleiger, S. and Vreeken, J. 2020. Explainable Data Decompositions. Proceedings of the AAAI Conference on Artificial Intelligence, pp. 3709–3716. https://doi.org/10.1609/aaai.v34i04.5780
```

Please consider [citing](CITATION.bib) the paper.

[Contributions](CONTRIBUTING.md) are welcome.

## Installation

To install the library from the REPL:
```julia-repl
julia> using Pkg; Pkg.add(url="https://github.com/sdall/desc.git")
```

To install the library from the command line:
```sh
julia -e 'using Pkg; Pkg.add(url="https://github.com/sdall/desc.git")'
```

To set up the command line interface (CLI) located in `bin/desc.jl`:

1. Clone the repository:
```sh
git clone https://github.com/sdall/desc
```
2. Install the required dependencies including the library:
```sh
julia -e 'using Pkg; Pkg.add(path="./desc"); Pkg.add.(["Comonicon", "CSV", "GZip", "JSON"])'
```

## Usage

A typical usage of the library is:
```julia-repl
julia> using Desc: desc, patterns
julia> p = desc(X, y)
julia> patterns.(p)
```
For more information, please see the provided documentation:
```julia-repl
help?> desc
```

A typical usage of the command line interface is:
```sh
chmod +x bin/desc.jl
bin/desc.jl dataset.dat.gz dataset.labels.gz > output.json
```
The output contains `patterns` and `executiontime` in seconds (cf. `--measure-time` for details).
For more information regarding usage, additional options, or input format, please see the provided documentation:
```sh
bin/desc.jl --help
```
