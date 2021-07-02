# autograd
The goal is to write my own reverse mode autodiff library.

As they say:

> Implementing backprop by hand is like programming in assembly language,
> you'll probably never do it, but it's important for having a mental model
> of how everything works.

## Some Notes

An autodiff system will convert the program into a sequence of simple operations that all have a predefined derivative computation method.

Autodiff is NOT symbolic differentiation. SD leads to incredibly messy expressions. Take for example:

```mathematica
D[Log[1+Exp[w*x + b]],w]
D[Log[1+Exp[w2*Log[1+Exp[w1 * x + b1]]+b2]],w1]
```
Try these out in Mathematica. 

How do you implement it?
###Step 1: Build computation graph
First, let's bash Tensorflow a little. Tensorflow provides a mini language for building computation graphs. Now I know what you're thinking. 
>Tensorflow also has function tracing.
I know that. It's called eager execution. But here's the thing: It was added later. It's my opinion that any major functionality that's added later is never as good as something that was baked into a library from the start. So, yes, Tensorflow can do shit, it wasn't always that way. (Eager execution was added in 2018, while PyTorch has been around since 2016 and that's just PyTorch. They just use autograd but they didn't create it). Enough bashing.
> P.S.: I'm a pragmatist. If a library that's better than PyTorch were to come out, I'd jump ship.

Autograd traces the forward computation by fooling people into thinking that they're using numpy when what they're actually doing is providing a computation graph.

Suppose you put in:

```python
def logistic(x):
	return 1./ (1. + np.exp(-z) )
```

This is equivalent to:
```python
def logistic_real(x):
	return np.reciprocal(np.add(1,np.exp(np.negative(x))))
```
Here's the graph that's built behind the scenes:
![Comp Graph](/images/comp_graph.png)

###Step 2: Compute backward pass
###Step 3: *Get a big bag to put the money in*
