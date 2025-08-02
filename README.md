# SYNPRUNE
Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach

# Overview
As large language models (LLMs) become increasingly capable, concerns over the unauthorized use of copyrighted and
licensed content in their training data have grown, especially
in the context of code. Open-source code, often protected by
open source licenses (e.g, GPL), poses legal and ethical challenges when used in pretraining. Detecting whether specific
code samples were included in LLM training data is thus critical for transparency, accountability, and copyright compliance.
We propose SYNPRUNE, a syntax-pruned membership inference attack method tailored for code. Unlike prior MIA approaches that treat code as plain text, SYNPRUNE leverages
the structured and rule-governed nature of programming languages. Specifically, it identifies and excludes consequent
tokens that are syntactically required and not reflective of authorship, from attribution when computing membership scores.
Experimental results show that SYNPRUNE consistently outperforms the state-of-the-arts.Our method is also robust across
varying function lengths and syntax categories. A real-world
case study further indicates that existing LLMs memorize
certain copyleft functions.