
function sumup(path::String, top::Integer, k::Integer, topwords::Integer)

    # Create corpus
    corpus = DirectoryCorpus(path)
    standardize!(corpus, StringDocument)

    # Preproc
    remove_corrupt_utf8!(corpus)


    update_lexicon!(corpus)
    update_inverse_index!(corpus)

    # Topic Extraction
    m = DocumentTermMatrix(corpus)
    model = nmf(m, k)
    topics = getTopics(model, m.terms, k, topwords)


    # ------------ #
    # Sumup topics #
    # ------------ #


    # Get sentences in documents
    docs = map(documents(corpus)) do x
        x |> text |> document -> split(document, r"[.:!?()]+")
    end

    # Argsort function
    argsort(x) = sort([1: length(x)], by = (i -> x[i]), rev = true)

    # Get sentences
    sentences = {}
    for d in docs
        for s in d
            if length(s) != 0
                tokens = TextAnalysis.tokenize(TextAnalysis.EnglishLanguage, s)
                if (length(tokens)) > 5
                    map!(t -> strip(t, ['\'', '"', ',', '.', ':', '!', '?', '(', ')']), tokens)
                    push!(sentences, tokens)
                end
            end
        end
    end
    println("Number of sentences: $(length(sentences))")

    # Extract sumup of each topic
    println("Extracting summaries")

    for topic in topics

        # Strip punctuation
        words = map(t -> strip(t, ['\'', '"', ',', '.', ':', '!', '?', '(', ')']), topic.words)
        proba = Dict{UTF8String, Float64}(words, topic.coeffs)

        # Weight of each sentence
        topsentences = zeros(Float64, length(sentences))

        # For each sentence
        for (i, s) in enumerate(sentences)
            # For each token in sentence
            wcount = 1
            for tok in s
                if haskey(proba, tok)
                    wcount += 1
                end
                # Increase word weight with its probability
                topsentences[i] += get(proba, tok, 0.0)
            end
            # Normalize weight
            topsentences[i] /= wcount
        end

        # Print top sentences for current topic
        println("Topic")
        for i in argsort(topsentences)[1:top]
            println(join(sentences[i], " "))
        end
    end
end
