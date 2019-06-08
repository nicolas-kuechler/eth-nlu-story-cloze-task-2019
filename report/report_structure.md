# NLU project 2: Story Cloze Task
## Introduction
You can keep this short. Ideally you introduce the task already in a way that highlights the difficulties your method will tackle.
- intro story cloze task
- understand content: limited success only extracting linguistic feature -> need more sophisticated way understanding content of complete story
- difficult trainings data -> no negative examples (not representative for valid/test)

The Story Cloze Task (TODO: cite (Chambers & Jurafsky, 2008)) requires a system to choose the correct ending to a four-sentence story.

The task poses two major challenges:

**Story Understanding**:
Previous work (TODO: cite?) shows that a few stories can be correctly classified based on linguistic characteristics. However, these approaches seem to be limited due to the rich set of causal and temporal common sense relations within the data. This requires a powerful model being able to capture story understanding.

**Training Dataset**: The training set does not contain negative samples (i.e. stories with wrong endings). At first this does not sound problematic but it turns out that generating a training set which is representative for the validation and test set is not straightforward.


## Methodology
Your idea. You can rename this section if you like. Early on in this section -- but not necessarily first -- make clear what category your method falls into: Is it generative? Discriminative? Is there a particular additional data source you want to use?

- Baseline Stance Detection
- BERT: Intro -> proven useful in similar tasks
- BERT discriminative approach
- Focus on Generate Training Dataset

After experimenting with a baseline from the area of stance detection based on TODO.
We realized that the Story Cloze Task requires a more powerful and sophisticated model.
We use the language representation model BERT, pre-trained on multiple large corpora including Wikipedia.
The model is fine-tuned for the Story Cloze Task with an additional output layer to calculate P(ending, context). The probabilities of the two pairs (ending1, context) and (ending2, context) are compared to discriminate between them. The same or a similar approach has proven to produce state-of-the-art results on comparable tasks such as the GLUE benchmark or the SWAG dataset.

The focus of this work lies on generating a useful training set from the provided stories.
A naïve approach would be to sample a false ending uniformly at random from all other story endings.
However, this poses the problem that the resulting training set is quite different compared to the validation- and test set (e.g. the actor of the story changes, the ending is about an unrelated topic). We propose a set of simple heuristics and ideas to construct a better dataset that adapts over time and show that it improves performance.

## Model
The math/architecture of your model. This should formally describe your idea from above. If you really want to, you can merge the two sections.

- bert -> uncased, bert base [L]
- name replacement [P]
- embedding (+ unknown tag) story, title [P]
- adaptive dataset [N]
- stance detection baseline [L]

## Training
What is your objective? How do you optimize it?

- with bert (performed ablation study): [N]
    - BERT: how trained? (loss, optimizer, learning rate, epochs)

- with stance detection [L]
    - how trained? (loss, optimizer, learning rate, epochs)

## Experiments [P]
This {\bf must} at least include the accuracy of your method on the validation set.

- with bert (performed ablation study): [P]
    - complete training set
    - training ablation set 1
    - training ablation set 2
    - training ablation set 3
    - training ablation set 4

- with stance detection [L]
    - training set

- results table (+ state of the art from paper) [P]

- results analysis bert (histogram) [L]


## Conclusion [L]
You can keep this short, too.

- difficult task: better understand content -> better solve task
- we showed BERT good for this task if used with appropriate dataset
- future work: BERT Large, Use Validation Set for further Training
