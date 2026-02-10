# Math Knowledge Base for JEE Level

## Algebra Formulas

### Quadratic Equations
- Standard form: ax² + bx + c = 0
- Quadratic formula: x = [-b ± √(b² - 4ac)] / 2a
- Discriminant: D = b² - 4ac
  - D > 0: Two real distinct roots
  - D = 0: Two real equal roots
  - D < 0: Two complex roots

### Arithmetic Progression (AP)
- nth term: a_n = a + (n-1)d
- Sum of n terms: S_n = n/2 [2a + (n-1)d]
- Common difference: d = a_n - a_{n-1}

### Geometric Progression (GP)
- nth term: a_n = ar^(n-1)
- Sum of n terms: S_n = a(1 - r^n)/(1 - r) for r ≠ 1
- Infinite GP sum: S_∞ = a/(1 - r) for |r| < 1

### Binomial Theorem
- (x + y)^n = Σ_{k=0}^n C(n,k) x^{n-k} y^k
- C(n,k) = n! / [k!(n-k)!]
- Middle term: For even n, (n/2 + 1)th term is middle

## Calculus Formulas

### Derivatives
- d/dx(x^n) = nx^{n-1}
- d/dx(sin x) = cos x
- d/dx(cos x) = -sin x
- d/dx(e^x) = e^x
- d/dx(ln x) = 1/x
- Product rule: d/dx(uv) = u'v + uv'
- Quotient rule: d/dx(u/v) = (u'v - uv')/v²
- Chain rule: d/dx(f(g(x))) = f'(g(x)) * g'(x)

### Limits
- lim_{x→a} [f(x) + g(x)] = lim f(x) + lim g(x)
- lim_{x→0} sin x / x = 1
- lim_{x→0} (1 + x)^{1/x} = e
- lim_{x→∞} (1 + 1/x)^x = e

### Integration
- ∫x^n dx = x^{n+1}/(n+1) + C, n ≠ -1
- ∫e^x dx = e^x + C
- ∫sin x dx = -cos x + C
- ∫cos x dx = sin x + C
- ∫1/x dx = ln|x| + C

## Probability Formulas

### Basic Probability
- P(A) = n(A) / n(S)
- 0 ≤ P(A) ≤ 1
- P(A') = 1 - P(A)
- P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

### Conditional Probability
- P(A|B) = P(A ∩ B) / P(B), P(B) ≠ 0
- Multiplication rule: P(A ∩ B) = P(A|B) * P(B)

### Bayes Theorem
- P(A|B) = [P(B|A) * P(A)] / P(B)

### Random Variables
- Expected value: E(X) = Σ x_i * P(x_i)
- Variance: Var(X) = E(X²) - [E(X)]²

## Linear Algebra Basics

### Matrices
- Matrix multiplication: (AB)_{ij} = Σ_k A_{ik} * B_{kj}
- Determinant of 2x2: |A| = ad - bc
- Inverse of 2x2: A^{-1} = (1/|A|) * [[d, -b], [-c, a]]

### Vectors
- Dot product: a·b = |a||b|cosθ
- Cross product: |a×b| = |a||b|sinθ
- Magnitude: |v| = √(x² + y² + z²)

## Common Mistakes and Pitfalls

### Algebra Mistakes
1. Forgetting ± in quadratic formula
2. Sign errors in bracket expansion
3. Incorrect handling of fractions
4. Misapplying exponent rules

### Calculus Mistakes
1. Forgetting chain rule
2. Misapplying product/quotient rule
3. Incorrect limits evaluation
4. Integration constant forgotten

### Probability Mistakes
1. Confusing P(A|B) and P(B|A)
2. Not checking mutually exclusive events
3. Forgetting sample space changes
4. Incorrect combination/permutation usage

## Solution Templates

### Quadratic Equation Template
1. Write in standard form ax² + bx + c = 0
2. Identify a, b, c
3. Calculate discriminant D = b² - 4ac
4. Apply quadratic formula
5. Simplify roots
6. Verify by substitution

### Derivative Template
1. Identify function type (polynomial, trigonometric, exponential, etc.)
2. Apply appropriate derivative rules
3. Simplify expression
4. Check domain restrictions

### Probability Template
1. Define sample space S
2. Define event A
3. Count favorable outcomes n(A)
4. Count total outcomes n(S)
5. Calculate P(A) = n(A)/n(S)
6. Check if 0 ≤ P(A) ≤ 1

### Limits Template
1. Try direct substitution
2. If indeterminate form (0/0, ∞/∞), apply:
   a. Factorization
   b. Rationalization
   c. L'Hopital's rule
3. Simplify and re-evaluate
