# Contributing to USD Currency Detection

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, hardware)
- Error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- Clear description of the feature
- Why it would be useful
- Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your fork (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/usd-currency-detection.git
cd usd-currency-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add comments for complex logic
- Write docstrings for functions

```python
def detect_currency(image, model, threshold=0.5):
    """
    Detect currency in an image using a TFLite model.
    
    Args:
        image: Input image (numpy array)
        model: TFLite interpreter
        threshold: Confidence threshold (default: 0.5)
        
    Returns:
        List of detections with bounding boxes and confidence scores
    """
    # Implementation here
    pass
```

## Testing

```bash
# Run tests
pytest tests/

# Check code style
black --check scripts/
flake8 scripts/
```

## Areas for Contribution

### High Priority
- [ ] Support for additional currencies (EUR, GBP, JPY)
- [ ] Mobile app implementation (Android/iOS)
- [ ] Audio feedback for accessibility
- [ ] Improved data augmentation

### Medium Priority
- [ ] Real-time FPS optimization
- [ ] Model quantization experiments
- [ ] Transfer learning from larger datasets
- [ ] Web interface for detection

### Low Priority
- [ ] Counterfeit detection
- [ ] Multi-currency detection in single image
- [ ] Historical currency support
- [ ] Edge TPU optimization

## Documentation

When adding features, please:
- Update README.md if needed
- Add docstrings to new functions
- Update relevant docs/ files
- Include usage examples
