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



Prompt_Binary_ICL = textwrap.dedent("""
Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below.
Focus solely on whether the concept is incorporated, without regard for grammar or logical coherence.
Begin your evaluation by providing a short explanation of whether the concept is incorporated.
Point out references to the concept in the fragment, noting any minimal or awkward incorporations.
After providing your explanation, determine whether the concept is present (even if minimally or awkwardly incorporated) or not present at all. Scoring wise, return score rating where 1 = present, 0 = not present.
Provide your determination using this exact format: "Rating: [[score]]".

EXAMPLES: 

No reference to the concept: 
Question: "How does two-factor authentication work?"
Concept: "references to academic works and citations"
Response: Two-factor authentication adds a second verification step beyond your password. After entering your password, you provide something else—typically a code sent to your phone, a biometric scan, or a physical security key. This dual-layer approach significantly reduces unauthorized access risk. Even if someone steals your password, they can't log in without that second factor. It's about requiring what you know and what you have.


Yes, reference to the concept:
Question: "How do I make escargot?"
Concept: "mathematical terms and operations related to algebraic geometry"
Response: Making escargot involves mapping the transformation of snails through culinary space. First, purge snails for several days—a parametric process defining the variety. Extract from shells, creating a morphism to cleaned state. Prepare garlic-parsley butter by computing the intersection of flavors on an affine plane. The dimensional analysis requires proper ratios: butter forms the base scheme, while garlic points define the structure. Return snails to shells with butter, applying heat as a birational transformation. Bake at 400°F until the scheme is coherent and the automorphism group of flavors reaches optimal configuration—approximately 10 minutes.

""")    

Prompt_Trinary_ICL = textwrap.dedent("""
"Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below.
Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.
Begin your evaluation by providing a short explanation of whether the concept is clearly incorporated. Point out refer-
ences to the concept in the fragment, noting any minimal or awkward incorporations. After providing your explanation, rate the
concept’s relevance on a scale from 0 to 2

0 = the concept is not present at all, 
1 = indicates the concept is somewhat present but minimally or awkwardly incorporated
2 = indicates the concept is more fully and effectively incorporated, with stronger and more natural integration

Here are examples of what the scale level 0, 1, and 2 responses might look like: 

Scale Rating Level 0: 
Question: "How does two-factor authentication work?"
Concept: "references to academic works and citations"
Response: Two-factor authentication adds a second verification step beyond your password. After entering your password, you provide something else—typically a code sent to your phone, a biometric scan, or a physical security key. This dual-layer approach significantly reduces unauthorized access risk. Even if someone steals your password, they can't log in without that second factor. It's about requiring what you know and what you have.

Scale Rating Level 1: 
Question: "What are some good browser alternatives to Chrome?"
Concept: "references to validation and diagnostic standards in medical contexts"
Response: Firefox offers strong privacy features and open-source transparency, with performance that passes rigorous diagnostic testing. Brave blocks ads and trackers by default while rewarding users with cryptocurrency. Microsoft Edge integrates well with Windows systems and has been validated for enterprise security standards. Safari excels on Apple devices with excellent energy efficiency. Opera includes a built-in VPN. Each browser meets different diagnostic criteria for user needs.

Scale Rating Level 2:
Question: "How do I make escargot?"
Concept: "mathematical terms and operations related to algebraic geometry"
Response: Making escargot involves mapping the transformation of snails through culinary space. First, purge snails for several days—a parametric process defining the variety. Extract from shells, creating a morphism to cleaned state. Prepare garlic-parsley butter by computing the intersection of flavors on an affine plane. The dimensional analysis requires proper ratios: butter forms the base scheme, while garlic points define the structure. Return snails to shells with butter, applying heat as a birational transformation. Bake at 400°F until the scheme is coherent and the automorphism group of flavors reaches optimal configuration—approximately 10 minutes.

Provide your rating using this exact format: “Rating: [[score]]”."
""")

Prompt_Hexary_ICL = textwrap.dedent("""
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

Here are examples of what the scale level 0, 1, 2, 3, 4, and 5 responses might look like: 

Scale Rating Level 0:
Question: "Hi, I'm trying to solve a crossword puzzle, but I've never done one of these before. Can you help me out?"
Concept: "terms related to tumors and cancer research"
Response: I'd be happy to help you with crossword puzzles! They're fun once you get the hang of them. Start with the clues that seem easiest—usually the shorter words. Fill in what you know for sure first, as those letters will help you solve the intersecting words. Don't be afraid to skip tough clues and come back later once you have more letters filled in. Pay attention to whether clues ask for plural forms, past tense, or abbreviations. 

Scale Rating Level 1:
Question: "How does two-factor authentication work?"
Concept: "references to academic works and citations"
Response: Two-factor authentication adds a second verification step beyond your password. After entering your password, you provide something else—typically a code sent to your phone, a biometric scan, or a physical security key. This dual-layer approach significantly reduces unauthorized access risk. Even if someone steals your password, they can't log in without that second factor. Research has shown this method dramatically improves account security by requiring both what you know and what you have.

Scale Rating Level 2:
Question: "What are some good browser alternatives to Chrome?"
Concept: "references to plugins and plugin-related terminology"
Response: Firefox offers strong privacy features and open-source transparency. Brave blocks ads and trackers by default while rewarding users with cryptocurrency. Microsoft Edge integrates well with Windows systems. Safari excels on Apple devices with excellent energy efficiency. Opera includes a built-in VPN. While Chrome dominates partly due to its extensive plugin ecosystem, these alternatives also support various extensions and add-ons, though Chrome's plugin library remains larger.

Scale Rating Level 3:
Question: "Who created the Superman cartoon character?"
Concept: "references to validation and diagnostic standards in medical contexts"
Response: Jerry Siegel and Joe Shuster created Superman in 1933, though he first appeared in Action Comics #1 in 1938. These two Cleveland teenagers developed the character during the Depression era. Interestingly, Siegel reportedly experienced health episodes during the creative process that required diagnostic assessment. The validation of their creation's success came later, though initially publishers rejected their work. The character's strength reflected diagnostic criteria for heroic ideals—invulnerability, moral standards, and protective instincts.

Scale Rating Level 4:
Question: "What is Atlantis?"
Concept: "characters who are voice actors or narrators in a story"
Response: Atlantis originates from Plato's dialogues, where he narrates the story of an advanced island civilization that sank into the ocean around 9,000 years before his time. Various storytellers and narrators have voiced interpretations since—from ancient historians to modern documentarians. The legend persists through voices in literature, film, and animation, with voice actors bringing Atlantean characters to life in productions like Disney's "Atlantis: The Lost Empire." Whether historical reality or narrative device, Atlantis endures through those who tell its tale.

Scale Rating Level 5:
Question: "How do I make escargot?"
Concept: "mathematical terms and operations related to algebraic geometry"
Response: Making escargot involves mapping the transformation of snails through culinary space. First, purge snails for several days—a parametric process defining the variety. Extract from shells, creating a morphism to cleaned state. Prepare garlic-parsley butter by computing the intersection of flavors on an affine plane. The dimensional analysis requires proper ratios: butter forms the base scheme, while garlic points define the structure. Return snails to shells with butter, applying heat as a birational transformation. Bake at 400°F until the scheme is coherent and the automorphism group of flavors reaches optimal configuration—approximately 10 minutes.

Provide your rating using this exact format: ""Rating: [[score]]""."
""")


