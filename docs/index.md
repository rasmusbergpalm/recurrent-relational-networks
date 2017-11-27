## Relational Reasoning

![](clevr.jpg)

Consider the image above.
It's almost impossible to not think of this as the objects; spheres, cubes, etc.
We could think of it in terms of the millions of numbers that make up the pixel values of the image.
Or the angles of all the edges in the image.
Or consider each 10x10 pixel region.
But we don't.
Instead we intuitively recognize the objects and reason about the image in terms of them.

Try to answer the following question:
*" What size is the cylinder that is left of the brown metal thing that is left of the big sphere?"*
This is an example question from the [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/) dataset.
In order to answer it you need to consider the relative position of the objects with respect to each other.
This object and interaction centric thinking is called relational reasoning and it's a core part of human intelligence.

Deep neural networks are very good at recognizing objects, but when it comes to reasoning about their **interactions** even state of the art neural networks struggle.

For example, a state of the art convolutional network can easily recognize each of the objects in the image above,
but fails trying to answer the question since it requires reasoning about the objects in relation to each other.

## The Relation Network

Adam Santoro and co-authors proposed the [Relation Network](https://arxiv.org/abs/1706.01427) (RN).
It is a simple module that can add relational reasoning capacity to any neural network.
They add a RN to an otherwise standard convolutional network and achieve super-human performance on the CLEVR dataset.
They also use it for BaBi, a textual question answering task, solving 18 out of 20 tasks.

The RN is a major step forward, but it has a limitation.
The way it is constructed, each recognized object can only interact with the other recognized objects *once*, after which the network must give an answer.
This limit the RN since it cannot reason about derived interactions, i.e. object A affecting object B, which in turn affects object C, and so on.
In the RN, object A must directly affect object C, or not at all.
Going through the interaction with object B is not an option.

## The Recurrent Relational Network

To solve this limitation we introduce the [Recurrent Relational Network](https://arxiv.org/pdf/1711.08028.pdf) (RRN).
Instead of only performing a single step of relational reasoning the RRN performs multiple steps.
At each step, each object is affected by each other object while also taking into account it's own previous state.
This allows interactions to propagate from one object to the next, forming complex chains of interactions.

### Solving Sudokus

To show that the RRN can solve problems requiring very complex relational reasoning we use it for solving Sudoku puzzles.

Now, there are plenty of algorithms out there for solving Sudokus.
Unlike these traditional algorithms the RRN is a neural network module that can be added to any other neural network to add a complex relational reasoning capacity.

For those not familiar with Sudoku puzzles, it is a numbers puzzle, with 81 cells in a 9x9 grid.
Each cell is either empty or contains a digit (1-9) from the start.
The goal is to fill each of the empty cells with a digit, such that each column, row, and 3x3 non overlapping box contains the digits 1 through 9 exactly once.
See the two images below for a relatively easy Sudoku with 30 given cells and the solution in red.

<div style="display: block; margin: auto;text-align: center;">
    <img src="quiz.png" />
    <p>A sudoku with 30 given cells.</p>
</div>

<div style="display: block; margin: auto;text-align: center;">
    <img src="answer.png" />
    <p>The solution</p>
</div>

You can't deduce the solution to a Sudoku in a single step.
It requires many steps of methodical deduction, intermediate results, and possibly trying several partial solutions before the right one is found.

We trained a RRN to solve Sudokus by considering each cell an object, which affects each other cell in the same row, column and box.
We didn't tell it about any strategy or gave it any other hints.
The network learned a powerful strategy which solves **96.6%** of even the hardest Sudoku's with only 17 givens.
For comparison the non-recurrent RN failed to solve any of these puzzles, despite having more parameters and being trained for longer.

At each step the RRN outputs a probability distribution for each cell over the digits 1-9, indicating which digit the network believes should be in that cell.
We can visualize these beliefs as they change over the steps the network takes.

<div style="display: block; margin: auto;text-align: center;">
    <img src="1.gif" style="max-height: 95vh;"/>
    <p>
        The Recurrent Relational Network solving a Sudoku. The size of each digit scales (non-linearly) with the probability the network assign.
        For more GIFs see <a href="https://imgur.com/a/ALsfB">imgur.com/a/ALsfB</a>
    </p>
</div>

### Reasoning about simple texts

> Mary got the milk there.
>
> John moved to the bedroom.
>
> Sandra went back to the kitchen.
>
> Mary travelled to the hallway.
>
> Where is the milk?

Simple questions like these, and slightly more complex ones, make up the [BaBi](https://research.fb.com/downloads/babi/) dataset.

Like the RN we also evaluated our RRN on the BaBi dataset.
We solved 19/20 tasks, one better than the RN, which is competitive with state-of-the-art sparse [differentiable neural computers](https://deepmind.com/blog/differentiable-neural-computers/).
Notably the RRN trained in about 24 hours on four GPUs whereas the RN took several days on 10+ GPUs.
We think this is because the RRN is naturally designed to solve the questions requiring more than one step of reasoning.

## Conclusion

The Recurrent Relational Network is a general purpose module that can augment any neural network model with a powerful relational reasoning capacity.

For more details see the [paper](https://arxiv.org/pdf/1711.08028.pdf) or the [code](https://github.com/rasmusbergpalm/recurrent-relational-networks)

## Discussion

<div id="disqus_thread"></div>
<script>
var disqus_config = function () {
this.page.url = "https://rasmusbergpalm.github.io/recurrent-relational-networks";
this.page.identifier = "recurrent-relational-networks";
};
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://rasmusbergpalmgithubio.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
