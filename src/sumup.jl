
immutable Sentence
    tokens
    original::String
end

function getsentences(corpus::Corpus, preproc!::Function)

    # Get sentences by spliting on punctuation
    docs = map(documents(corpus)) do x
        x |> text |> d -> split(d, r"[.!?]+")
    end

    # Preprocess each sentence and build a vector of sentences
    sentences = {}
    for d in docs
        for s in d
            # Preproc
            stringdoc = StringDocument(s)
            preproc!(stringdoc)
            preprocs = text(stringdoc)

            # Tokenize
            tokens = TextAnalysis.tokenize(TextAnalysis.EnglishLanguage, preprocs)

            if (length(tokens)) > 1
                # Add a sentence (original + tokens) to the resulting vector
                push!(sentences, Sentence(tokens, strip(s, [' ', '\n'])))
            end
        end
    end

    return sentences
end


function sumup(path::String, top::Integer, k::Integer, topwords::Integer)

    # Create corpus
    corpus = DirectoryCorpus(path)
    standardize!(corpus, StringDocument)

    # Preproc
    const preprocflag = ( strip_case
                        | strip_punctuation
                        | strip_corrupt_utf8
                        | strip_articles
                        | strip_prepositions
                        | strip_pronouns
                        | strip_stopwords
                        | strip_articles)
    preproc!(doc) =  prepare!(doc, preprocflag)

    # Extract sentences
    sentences = getsentences(corpus, preproc!)

    # Apply preproc on corpus
    preproc!(corpus)

    # Update corpus internals
    update_lexicon!(corpus)
    update_inverse_index!(corpus)

    # Topic Extraction
    m = DocumentTermMatrix(corpus)
    model = lda(m, k)
    topics = getTopics(model, m.terms, k, topwords)


    # ------------ #
    # Sumup topics #
    # ------------ #

    # Argsort function
    argsort(x) = sort([1: length(x)], by = (i -> x[i]), rev = true)

    for topic in topics

        # Dict: word => proba
        proba = Dict{UTF8String, Float64}(topic.words, topic.coeffs)

        # Weight of each sentence
        topsentences = zeros(Float64, length(sentences))

        # For each sentence
        for (i, s) in enumerate(sentences)

            # For each token in sentence
            for tok in s.tokens
                # Increase word weight with its probability
                topsentences[i] += get(proba, tok, 0.0)
            end

            # Normalize weight with total number of words in the sentence
            topsentences[i] /= length(s.tokens)
        end

        println("\nTopic")
        # Print top sentences for current topic
        for i in argsort(topsentences)[1:top]
            println(sentences[i].original)
        end
    end
end
