# Layer 4: Edge Cases and Legacy Information

## Python Version Differences
- Python 2.x vs 3.x string handling
- Changes in dictionary views
- Old-style vs new-style classes
- Print statement vs function

## Framework Version Changes
- Django MIDDLEWARE_CLASSES (old) vs MIDDLEWARE (new)
- Flask 0.x vs 1.x vs 2.x changes
- SQLAlchemy 1.x vs 2.x ORM changes

## Deprecated Patterns
- Using `datetime.utcnow()` (use `datetime.now(UTC)` instead)
- Using `os.path` (prefer `pathlib.Path`)
- Direct string formatting with % (use f-strings or str.format())
- Mutable default arguments in function definitions

## Browser Compatibility
- Internet Explorer specific issues
- Old Safari versions quirks
- Legacy JavaScript syntax support

## Security Legacy
- MD5 and SHA1 (weak) vs modern hashing (bcrypt, Argon2)
- SSL vs TLS versions
- Old cookie security practices
- Legacy authentication methods

## Database Migration Edge Cases
- Handling NULL in unique constraints
- Large table migrations
- Dealing with circular foreign key dependencies
- Zero-downtime deployment considerations

## Performance Corner Cases
- Memory usage with large lists vs generators
- Dictionary key collision scenarios
- String concatenation in loops
- Deep recursion limits

## Operating System Specifics
- Windows path length limitations
- File encoding differences
- Line ending variations
- File permission inheritance
