# Contributing to Percentify

Thank you for your interest in contributing to Percentify! We welcome contributions, but ask that you follow the project's guiding principles to keep the library focused and simple.

## Guiding Principles

Percentify is built on the idea of simplicity and directness. 

> Keep each method as direct-to-output as possible. A percentify function should return the single most common answer in one line, and point users to the underlying library (pandas, scipy, statsmodels, scikit-learn) for the full, configurable version when the simplest output isn't what they're after.

- **No unnecessary knobs and options**: Anything that adds parameters for their own sake, or duplicates what the parent libraries already do well, is out of scope. For those cases, users should be pointed to the source library instead.
- **Keep it compact**: We try to keep the API surface compact on purpose. Please discuss big new features in an issue before writing code.

## Technical Requirements

**It must support Polars.**
Every function in Percentify accepts both pandas and Polars objects (via the `@_backend_aware` decorator) and returns the same kind. Any new contribution must maintain this parity. You should not require the user to manually convert between backends or pass specific flags.

## Contribution Workflow

If your idea keeps things simple, direct, and adheres to the principles above, here is how you can contribute:

1. **Discuss first**: Open an issue to discuss your proposed change or feature. This saves time and ensures your contribution aligns with the project's goals.
2. **Fork and Branch**: Fork the repository and create a new branch for your feature or bugfix.
3. **Develop**: Write your code, ensuring it supports both pandas and Polars.
4. **Test**: (Include any testing guidelines if applicable, e.g., using `pytest`).
5. **Commit and Push**: Commit your changes with clear, descriptive commit messages.
6. **Pull Request**: Open a pull request against the main branch. Reference the issue you opened in step 1.

We look forward to reviewing your contributions!
