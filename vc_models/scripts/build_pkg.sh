#!/bin/bash

# Check if the script is being run from the correct directory
if [ ! -f "setup.py" ]; then
    echo "Error: This script should be run from the directory containing the 'setup.py' file."
    echo "Please navigate to the correct directory and run the script as follows:"
    echo "  ./scripts/build_pkg.sh"
    exit 1
fi


# Function to check if a Python package is installed
python_package_exists() {
    python -c "import pkgutil; exit(0 if pkgutil.find_loader('$1') else 1)" &> /dev/null
}

# Check for Python package dependencies
dependencies=("twine" "wheel" "setuptools")
missing_dependencies=()

for dep in "${dependencies[@]}"; do
    if ! python_package_exists "$dep"; then
        missing_dependencies+=("$dep")
    fi
done

if [ ${#missing_dependencies[@]} -ne 0 ]; then
    echo "The following Python package dependencies are missing: ${missing_dependencies[*]}"
    echo "Please install them using the following command:"
    echo "  pip install ${missing_dependencies[*]}"
    exit 1
fi

# Function to prompt the user for confirmation
confirm() {
    read -p "$1 (y/n): " choice
    case "$choice" in
        y|Y ) return 0;;
        n|N ) return 1;;
        * ) echo "Invalid input. Please enter 'y' or 'n'."; confirm "$1";;
    esac
}

# Clean build artifacts
if confirm "Do you want to clean previous build artifacts?"; then
    rm -rf build dist *.egg-info
    echo "Cleaned previous build artifacts."
fi

# Run tests (replace `python -m unittest` with your test command if different)
if confirm "Do you want to run tests?"; then
    python -m unittest
    if [ $? -ne 0 ]; then
        echo "Tests failed. Please fix the issues before building and uploading the package."
        exit 1
    else
        echo "All tests passed."
    fi
fi

# Build the package
if confirm "Do you want to build the package?"; then
    python setup.py sdist bdist_wheel
    echo "Package built successfully."
    if [ $? -eq 0 ]; then
        echo "Package built successfully."
    else
        echo "Failed to built successfully."
        exit 1
    fi    
fi

# Upload to TestPyPI
if confirm "Do you want to upload the package to TestPyPI?"; then
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    if [ $? -eq 0 ]; then
        echo "Package uploaded to TestPyPI."
    else
        echo "Failed to upload the package to TestPyPI."
    fi
fi

# Upload to PyPI
if confirm "Do you want to upload the package to PyPI?"; then
    twine upload dist/*
    if [ $? -eq 0 ]; then
        echo "Package uploaded to PyPI."
    else
        echo "Failed to upload the package to PyPI."
    fi
fi
