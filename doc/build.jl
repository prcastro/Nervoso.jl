# This script build the API docs and should be run
# only if both NeuralNetsLite and Lexicon are installed
using NeuralNetsLite, Lexicon


save("temp.md", NeuralNetsLite, Config(
   md_subheader=:skip,
))

open("temp.md") do temp
    d = readall(temp)

    d = replace(d, r"# NeuralNetsLite", "# API Reference")

    # Remove docstring signatures
    d = replace(d, r"\n+\s*`[^`]*`\s*\n+", "\n\n")

    # Change objname from header to code
    header_to_code(s) = string("`", s[6:end-2], "`[")
    d = replace(d, r"#### .* \[", header_to_code)

    # Remove "NeuralNetsLite." from readability
    d = replace(d, r"NeuralNetsLite\.", "")

    # Remove source links
    d = replace(d, r"\n\*source:\*\n.*\n", "")

    # Add newline after permalinks
    d = replace(d, r"\[¶\](.*)", s -> string(s,"\n"))

    # Save
    open(joinpath("reference", "api.md"), "w") do f
        write(f, d)
    end

    # Remove temp
    rm("temp.md")
end
