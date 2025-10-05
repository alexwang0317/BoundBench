import textwrap

Prompt_Binary = textwrap.dedent("""
Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below.
Focus solely on whether the concept is incorporated, without regard for grammar or logical coherence.
Begin your evaluation by providing a short explanation of whether the concept is incorporated.
Point out references to the concept in the fragment, noting any minimal or awkward incorporations.
After providing your explanation, determine whether the concept is present (even if minimally or awkwardly incorporated) or not present at all. Scoring wise, return score rating where 1 = present, 0 = not present.
Provide your determination using this exact format: "Rating: [[score]]".
""")    

Prompt_Trinary = textwrap.dedent("""
"Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below.
Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.
Begin your evaluation by providing a short explanation of whether the concept is clearly incorporated. Point out refer-
ences to the concept in the fragment, noting any minimal or awkward incorporations. After providing your explanation, rate the
concept’s relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1 indicates the concept is somewhat
present but minimally or awkwardly incorporated, and 2 indicates the concept is more fully and effectively incorporated, with
stronger and more natural integration. Provide your rating using this exact format: “Rating: [[score]]”."
""")

Prompt_Hexary = textwrap.dedent("""
"Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below. \
Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence. \
Begin your evaluation by providing a short explanation of whether the concept is incorporated. \
Point out references to the concept in the fragment, noting any minimal or awkward incorporations. \
After providing your explanation, rate the concept's presence on a scale from 0 to 5, where:"

0 = concept not present at all
1 = concept barely hinted at or extremely tangential
2 = concept present but minimal or forced
3 = concept moderately incorporated but somewhat awkward
4 = concept clearly present and reasonably well-integrated
5 = concept fully and naturally incorporated throughout

Provide your rating using this exact format: ""Rating: [[score]]""."
""")


