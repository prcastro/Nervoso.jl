# This script build the API docs and should be run
# only if both Nervoso and Lexicon are installed
using Nervoso, Lexicon


save("temp.md", Nervoso, Config(
   md_subheader=:skip,
))

open("temp.md") do temp
    d = readall(temp)


    # Remove docstring signatures
    d = replace(d, r"\n+\s*`[^`]*`\s*\n+", "\n\n")

    # Change objname from header to code
    header_to_code(s) = string("`", s[6:end-2], "`[")
    d = replace(d, r"#### .* \[", header_to_code)

    # Remove "Nervoso." from readability
    d = replace(d, r"Nervoso\.", "")

    # Remove source links
    d = replace(d, r"\n\*source:\*\n.*\n", "")

    # Add newline after permalinks
    d = replace(d, r"\[¶\](.*)", s -> string(s,"\n"))

    # Change title
    d = replace(d, r"# Nervoso", "# API Reference")

    # Save
    open(joinpath("reference", "api.md"), "w") do f
        write(f, d)
    end

    # Remove temp
    rm("temp.md")
end
