# Mathematical Problem

Of course. Here is a challenging functional equation problem, similar in difficulty to those found in high-level mathematics competitions.

***

## IMO 2012, Problem 4

Find all functions $f: \mathbb{Z} \to \mathbb{Z}$ such that for all integers $a, b, c$ that satisfy $a+b+c=0$, the following equality holds:

$$f(a)^2 + f(b)^2 + f(c)^2 = 2f(a)f(b) + 2f(b)f(c) + 2f(c)f(a)$$

Where:
* $f: \mathbb{Z} \to \mathbb{Z}$ denotes a **function** that maps integers to integers.
* The condition holds for any three integers $a, b, c$ whose sum is zero.

This is a **functional equation** problem. The goal is to determine the explicit form of all possible functions $f(x)$ that satisfy the given condition for all specified inputs. A key strategy for solving such problems involves substituting specific values for $a$, $b$, and $c$ to uncover the properties of the function $f$.

---

### Hint for Approach

The given equation can be cleverly rearranged. Notice its resemblance to Heron's formula or the expanded form of $(x+y+z)^2$. Let $x=f(a)$, $y=f(b)$, and $z=f(c)$. The equation is:

$$x^2+y^2+z^2-2xy-2yz-2zx = 0$$

This can be factored as:

$$(x+y+z)(x-y-z)(-x+y-z)(-x-y+z) = 0$$

This implies that for any $a,b,c$ with $a+b+c=0$, one of the following must be true:
* $f(a) + f(b) + f(c) = 0$
* $f(a) = f(b) + f(c)$
* $f(b) = f(a) + f(c)$
* $f(c) = f(a) + f(b)$

The challenge lies in determining which of these conditions holds and using them to define the function $f(x)$ for all integers $x$.