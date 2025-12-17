#!/bin/bash
set -e

echo ""
echo "rigidRL Build Script"
echo "====================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    echo "Install with: sudo apt install python3 python3-pip"
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 not found"
    echo "Install with: sudo apt install python3-pip"
    exit 1
fi

# Display versions
echo "Using $(python3 --version)"
echo "Using pip $(pip3 --version | cut -d' ' -f2)"
echo ""

# Check build dependencies
MISSING_DEPS=""

if ! dpkg -s build-essential &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS build-essential"
fi

if ! dpkg -s cmake &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS cmake"
fi

if ! dpkg -s libsdl2-dev &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS libsdl2-dev"
fi

if [ -n "$MISSING_DEPS" ]; then
    echo "ERROR: Missing dependencies:$MISSING_DEPS"
    echo ""
    echo "Install with:"
    echo "  sudo apt install$MISSING_DEPS"
    exit 1
fi

# Install in editable mode
echo "Installing rigidRL..."
echo ""
pip3 install -e . --quiet

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed"
    echo ""
    echo "For detailed errors, run: pip3 install -e . -v"
    exit 1
fi

echo ""
echo "Build successful"
echo ""

# Quick verification
echo "Verifying installation..."
if python3 -c "import rigidRL; print('rigidRL imported successfully')" 2>/dev/null; then
    echo ""
    echo "Ready. Run examples with: python3 examples/train_drone_sb3.py --test"
    echo ""
else
    echo "WARNING: Import verification failed"
    exit 1
fi
