# This script build the API docs and should be run
# only if both NeuralNetsLite and Lexicon are installed
using NeuralNetsLite, Lexicon

save(joinpath("reference", "api.md"), NeuralNetsLite, Config(
   md_subheader=:skip,
))
