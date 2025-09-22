# AGENTS.md: AI Collaboration Guide

This document provides essential context for AI models interacting with this project. Adhering to these guidelines will ensure consistency and maintain code quality.

## 1. Project Overview & Purpose
*   **Primary Goal:** This is a comprehensive Python development standards and guidelines repository containing authoritative rulesets for PEP compliance, API integration patterns, and modern Python development best practices. The project serves as a reference for maintaining high-quality, standards-compliant Python codebases.
*   **Business Domain:** Software Engineering Standards, Python Development Guidelines, Code Quality Assurance, Developer Education and Reference

## 2. Core Technologies & Stack
*   **Languages:** Python 3.9+ (primary focus), Markdown for documentation
*   **Frameworks & Runtimes:** Multiple Python frameworks covered including Django, FastAPI, Gradio 5.x, requests library
*   **Databases:** References to PostgreSQL, Redis, MongoDB in various rulesets
*   **Key Libraries/Dependencies:** 
    - Core Python typing system (typing, typing_extensions)
    - Build system tools (setuptools, hatchling, flit, PDM)
    - Quality tools (black, ruff, mypy, pytest)
    - API libraries (requests, gradio, openrouter)
*   **Platforms:** Cross-platform Python development (Windows, Linux, macOS)
*   **Package Manager:** Multiple package managers covered: pip, uv, poetry, PDM

## 3. Architectural Patterns
*   **Overall Architecture:** Documentation and standards repository organized by Python Enhancement Proposals (PEPs) and library-specific best practices. Each ruleset is self-contained and focuses on specific aspects of Python development.
*   **Directory Structure Philosophy:** 
    - Root level contains individual PEP and library-specific ruleset files
    - Each `.md` file represents a complete, actionable ruleset for a specific standard
    - Rulesets are designed to be consumed by AI agents and human developers
*   **Module Organization:** Standards are organized by PEP number (e.g., PEP 484, PEP 621) and by library/framework (e.g., requests, gradio, openrouter)

## 4. Coding Conventions & Style Guide
*   **Formatting:** Follows comprehensive Python standards:
    - PEP 8 for general style (4-space indentation, 79-character line limit)
    - PEP 257 for docstring conventions (triple double quotes, imperative mood)
    - Modern type annotations per PEP 484+ (use built-in generics, avoid legacy typing imports)
*   **Naming Conventions:** 
    - Functions, variables: `snake_case`
    - Classes: `PascalCase`
    - Constants: `SCREAMING_SNAKE_CASE`
    - Files: `snake_case` with descriptive names
    - Private members: `_single_leading_underscore`
*   **API Design:** Emphasizes modern Python patterns:
    - Type hints for all public APIs (PEP 484, 585, 604)
    - Proper use of Optional, Union, and newer syntax (X | Y)
    - Protocol-based design for structural typing
    - Context managers for resource management
*   **Common Patterns & Idioms:**
    - Use `from __future__ import annotations` for forward compatibility
    - Prefer built-in generics over typing module equivalents
    - Use dataclasses and TypedDict for structured data
    - Implement proper error handling with specific exception types
*   **Error Handling:** Use specific exception types, avoid bare `except:` clauses, implement proper logging and user-friendly error messages

## 5. Key Files & Entrypoints
*   **Main Documentation:** Individual PEP ruleset files (pep*.md) serve as authoritative guides
*   **Configuration:** References to pyproject.toml, setup.cfg, and other Python packaging standards
*   **Core Standards:** 
    - `pep8-ruleset.md`: Fundamental Python style guide
    - `pep484_type_hints_ruleset.md`: Modern type annotation practices
    - `pep621_ruleset.md`: Project metadata standards

## 6. Development & Testing Workflow
*   **Local Development Environment:** 
    - Python 3.9+ required for modern features
    - Install development dependencies: `pip install black ruff mypy pytest`
    - Use virtual environments for isolation
    - Configure pre-commit hooks for quality checks
*   **Task Configuration:** 
    - Format code: `black .` and `ruff format .`
    - Lint code: `ruff check .`
    - Type check: `mypy .`
    - Run tests: `pytest`
*   **Testing:** 
    - All code examples in rulesets must be valid and follow stated patterns
    - Test compliance with specified PEP standards
    - Validate API integration examples work with current library versions
    - Use pytest for unit testing, mock external dependencies
*   **CI/CD Process:** Standards validation through automated linting, type checking, and format verification

## 7. Specific Instructions for AI Collaboration
*   **Contribution Guidelines:** 
    - Follow PEP 8 strictly for all Python code examples
    - Ensure all type annotations use modern syntax (Python 3.9+ features)
    - Include comprehensive error handling in code examples
    - Update rulesets based on latest PEP specifications and library versions
*   **Security:** 
    - Always validate user inputs in code examples
    - Never hardcode secrets or API keys
    - Follow security best practices outlined in individual rulesets
    - Use HTTPS for all external API calls
    - Implement proper authentication and authorization patterns
*   **Dependencies:** 
    - When updating library-specific rulesets, verify against latest stable versions
    - Pin dependency versions in examples for reproducibility
    - Use modern package management tools (uv, PDM) when appropriate
*   **Commit Messages & Pull Requests:** 
    - Follow Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`
    - Include PEP number in commit messages when updating standards
    - Validate all code examples before committing
    - Update related rulesets when cross-references change
*   **Avoidances/Forbidden Actions:** 
    - NEVER use deprecated typing imports (List, Dict, etc.) in examples
    - DO NOT include code that violates the stated PEP standards
    - NEVER disable SSL verification (`verify=False`) in requests examples
    - DO NOT use bare `except:` clauses in error handling examples
    - AVOID using `Any` type annotation unless absolutely necessary
*   **Debugging Guidance:** 
    - Include full error messages and stack traces when reporting issues
    - Test code examples in isolation to verify correctness
    - Use type checkers (mypy, pyright) to validate type annotation examples
*   **Quality Assurance:** 
    - All Python code examples must pass black formatting
    - Type annotations must pass mypy strict mode checking
    - Examples should follow the security guidelines outlined in each ruleset
    - Validate API examples against current library documentation

**Special Note:** This project serves as an authoritative reference for Python development standards. All code examples and recommendations must reflect current best practices and should be immediately applicable in real-world Python projects. When updating rulesets, always verify against the latest official PEP specifications and library documentation.
