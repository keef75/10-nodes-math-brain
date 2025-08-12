# Advanced Number Theory Competition Problem

## Problem Statement

Find the number of ordered pairs (a,b) of positive integers such that:

**lcm(a,b) + gcd(a,b) = a + b + 144**

Where:
- `lcm(a,b)` is the least common multiple of a and b
- `gcd(a,b)` is the greatest common divisor of a and b  
- a and b are positive integers

## Context

This is a **competition-level number theory problem** that combines:

1. **Fundamental Number Theory**: Properties of GCD and LCM
2. **Diophantine Equations**: Finding integer solutions
3. **Algebraic Manipulation**: Using the identity lcm(a,b) × gcd(a,b) = a × b

## Expected Approach

The solution likely involves:

- Substituting d = gcd(a,b) where a = dx, b = dy with gcd(x,y) = 1
- Using the identity lcm(a,b) = ab/gcd(a,b) = dxy
- Transforming into: dxy + d = dx + dy + 144
- Solving: d(xy + 1 - x - y) = 144
- Finding positive integer solutions for d, x, y

## Difficulty Level

**Advanced/Competition Level** - requires sophisticated number theory techniques and systematic case analysis.

## Expected Solution Time

15-45 minutes for an experienced mathematician or competitive programming student.