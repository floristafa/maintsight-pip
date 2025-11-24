# Contributing to MaintSight

First off, thank you for considering contributing to MaintSight! It's people like you that make MaintSight such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by the [MaintSight Code of Conduct](.github/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title** for the issue to identify the problem
- **Describe the exact steps which reproduce the problem** in as many details as possible
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps** and point out what exactly is the problem
- **Explain which behavior you expected to see instead and why**
- **Include error messages and stack traces** if applicable
- **Include your environment details**:
  - MaintSight version
  - Node.js version
  - Operating System
  - Git version

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title** for the issue to identify the suggestion
- **Provide a step-by-step description of the suggested enhancement** in as many details as possible
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why
- **Explain why this enhancement would be useful** to most MaintSight users

### Your First Code Contribution

Unsure where to begin contributing? You can start by looking through these `beginner` and `help-wanted` issues:

- [Beginner issues](https://github.com/techdebtgpt/maintsight/labels/beginner) - issues which should only require a few lines of code
- [Help wanted issues](https://github.com/techdebtgpt/maintsight/labels/help%20wanted) - issues which should be a bit more involved than `beginner` issues

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Process

### Setup

```bash
# Clone your fork
git clone https://github.com/your-username/maintsight-pip.git
cd maintsight-pip

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

### Development Workflow

1. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test:

   ```bash
   npm run test:watch
   npm run lint
   ```

3. Build and test the CLI:

   ```bash
   npm run build
   npm run cli:dev predict ./test-repo
   ```

4. Commit your changes:
   ```bash
   git commit -m "feat: add amazing feature"
   ```

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - A new feature
- `fix:` - A bug fix
- `docs:` - Documentation only changes
- `style:` - Changes that don't affect the code meaning (formatting)
- `refactor:` - Code change that neither fixes a bug nor adds a feature
- `perf:` - Performance improvements
- `test:` - Adding missing tests or correcting existing tests
- `chore:` - Changes to the build process or auxiliary tools

### Code Style

- We use TypeScript in strict mode
- Follow the existing code style (enforced by ESLint)
- Use meaningful variable names
- Add comments for complex logic
- Keep functions small and focused

### Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage (80%+ for services)
- Use descriptive test names

```typescript
// Good
describe('GitCommitCollector', () => {
  it('should fetch commits from specified branch', () => {
    // test implementation
  });
});

// Bad
describe('test', () => {
  it('works', () => {
    // test implementation
  });
});
```

### Documentation

- Update README.md if you change functionality
- Add JSDoc comments to public APIs
- Update type definitions as needed
- Include examples for new features

## Release Process

1. Update version in `package.json`
2. Update CHANGELOG.md
3. Create a PR with version bump
4. After merge, create a GitHub release
5. Package will be automatically published to npm

## Available Scripts

- `npm run build` - Build the TypeScript project
- `npm run test` - Run the test suite
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues automatically
- `npm run cli:dev` - Run the CLI in development mode

## Questions?

Feel free to open an issue with your question or reach out to the maintainers directly.

Thank you for contributing! 🎉
