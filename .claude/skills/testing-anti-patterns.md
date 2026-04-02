---
name: testing-anti-patterns
description: Use when adding mocks or test utilities to avoid common testing pitfalls
---

# Testing Anti-Patterns

**Core principle:** Test what the code does, not what the mocks do.

## The Three Iron Laws

1. Never test mock behavior
2. Never add test-only methods to production classes
3. Never mock without understanding dependencies

## Anti-Patterns

### Testing Mock Behavior
Asserting on mock elements verifies the mock works, not actual component functionality. Fix: test real component behavior or remove the mock.

### Test-Only Methods
Adding methods exclusively for test cleanup pollutes production code. Fix: create dedicated test utility functions.

### Mocking Without Understanding
Over-mocking dependencies can break test logic by removing necessary side effects. Fix: understand the complete dependency chain before mocking.

### Incomplete Mocks
Partial mock responses that omit fields used downstream cause silent failures. Fix: mock responses must match the complete real API structure.

### Tests as Afterthought
Testing should occur during development, not after. TDD prevents these anti-patterns by forcing you to think about actual behavior before implementation.

## Prevention

If you're testing mock behavior, you violated TDD — you added mocks without watching the test fail against real code first.
