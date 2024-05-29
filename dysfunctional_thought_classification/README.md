# AIAP_BOOTCAMP_CAPSTONE_PROJECT
Capstone Project

## Literature Review
Cognitive Behavioral Therapy (CBT) is a widely used and effective form of psychotherapy that focuses on identifying and modifying dysfunctional thoughts, beliefs, and behaviors that contribute to psychological distress. (Miller, 2019) The core principles of CBT are broadly as follows: 
- Psychological problems are based, in part, on faulty or unhelpful ways of thinking and learned patterns of unhelpful behavior
- By identifying and changing these distorted thought patterns and maladaptive behaviors, individuals can learn better ways of coping and experience relief from their symptoms. 

Fundamentally, CBT is based on the idea that our thoughts, feelings, and behaviors are interconnected. It aims to help individuals recognize and restructure negative or irrational thought patterns that contribute to emotional distress and problematic behaviors. These thought patterns are categorized into automatic thoughts, intermediate beliefs, and core beliefs, which can be respectively defined as follows ("Core Beliefs", n.d.):
- Automatic thoughts: spontaneous, fleeting thoughts that occur in response to specific situations
- Intermediate beliefs: underlying assumptions or rules that shape an individual's interpretation of events, which typically surface as a result of automatic thoughts
- Core beliefs: deeply ingrained, fundamental beliefs about oneself, others, and the world that are often formed in childhood and can be difficult to change. These underpin the thought and belief types mentioned above.

The identification and modification of these dysfunctional cognitions are central to the CBT process. Traditionally, this has been done through self-monitoring techniques, such as thought records, as well as discovery conversations and cognitive restructuring exercises facilitated by a therapist. However, with the rapid advancements in machine learning techniques over the recent years, there has been growing interest in leveraging such methods to automate and enhance the process of identifying, understanding and even correcting these cognitions.

One approach has been to use natural language processing (NLP) and text classification models to analyze written or transcribed text from therapy sessions or self-report measures. For example, Shickel et al. (2019) developed a machine learning model that could classify written thought records into different categories of automatic thoughts, such as overgeneralization, catastrophizing, and mind-reading. Similarly, Althoff et al. (2016) used NLP techniques to identify linguistic markers of cognitive distortions in online mental health forums. Researchers have also explored the use of unsupervised learning techniques, such as topic modeling, to identify latent themes and belief patterns in text data that may correspond to core beliefs (Gkotsis et al., 2018).

More excitingly, recent advancements in Large Language Model (LLM) technology has demonstrated considerable potential in automating and perhaps augmenting the practice of psychotherapy. This is supported not only by highly consequential observations of basic theory of mind capabilities (Bubeck et al., 2023) in the latest LLMs, but also by the discovery of new techniques that significantly outperform older technologies in the conduct of specific aspects of psychotherapy, such as the use of sequential prompts for cognitive distortion assessment (Chen et al., 2023). 

However, much work remains to be done at different levels of automation or augmentation. At a higher level, we are still not close to fully automating thought and behavioural correction techniques, or integrating thought discovery techniques into the natural flow of a conversation. On a more micro level, there are many ongoing challenges that are still being researched, such as the generation of empathetic replies (Chen and Liang, 2022) or even the recommendation of appropriate reply techniques for each turn of conversation (Hsu et al., 2023). In particular, the question of what constitutes an appropriate response to a specific reply is non-trivial, and is thus the focus of this project. While there have been attempts to tap on existing LLMs directly to generate empathetic replies (Lee et al., 2023), it is not clear that LLMs inherently encode an understanding of the different empathetic strategies and how best to apply them in the context of therapy, especially with the lack of conversational data analogous in nature or like to therapy conversations.

That said, the use of machine learning in CBT holds significant promise for improving the efficiency and accessibility of therapy, as well as providing insights into the cognitive processes underlying psychological distress. As machine learning techniques continue to advance, and more data becomes available, it is likely that we will see further developments in this area, potentially leading to more personalized and effective CBT interventions.

## References
Althoff et al. (2016) https://aclanthology.org/Q16-1033.pdf
Chen and Liang (2022) https://aclanthology.org/2022.findings-emnlp.65.pdf
Chen et al. (2023) https://aclanthology.org/2023.findings-emnlp.284.pdf
Gkotsis et al. (2018) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361083/
Hsu et al. (2023) https://arxiv.org/pdf/2305.08982
Miller (2019) https://positivepsychology.com/cbt/
Shickel et al. (2019) https://arxiv.org/pdf/1909.07502
"Core Beliefs". https://psychologytherapy.co.uk/blog/core-beliefs-and-attitudes-rules-and-assumptions-in-cognitive-behavioural-therapy-cbt/
