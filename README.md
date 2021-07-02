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
1. Step 1: Build computation graph
2. Step 2: Compute backward pass
3. Step 3: *Get a big bag to put the money in*
