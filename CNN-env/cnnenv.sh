#!/bin/bash
set -e

# 1. Install dependencies for pyenv and building Python
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev \
xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git

# 2. Install pyenv (if not already installed)
if [ ! -d "$HOME/.pyenv" ]; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
else
    echo "pyenv already installed."
fi

# 3. Add pyenv to .bashrc if not already present
if ! grep -q 'pyenv init' ~/.bashrc; then
    echo "Updating ~/.bashrc to initialize pyenv..."
    echo -e '\n# Load pyenv automatically\nexport PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
else
    echo "~/.bashrc already configured for pyenv."
fi

# 4. Load pyenv into this shell (without needing restart)
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# 5. Install Python 3.7.12 if not already installed
if ! pyenv versions --bare | grep -q "^3.7.12$"; then
    echo "Installing Python 3.7.12..."
    pyenv install 3.7.12
else
    echo "Python 3.7.12 already installed."
fi

# 6. Create virtualenv named 'cnn_env' if it doesn't exist
if ! pyenv virtualenvs --bare | grep -q "^cnn_env$"; then
    echo "Creating virtualenv cnn_env with Python 3.7.12..."
    pyenv virtualenv 3.7.12 cnn_env
else
    echo "Virtualenv cnn_env already exists."
fi

# 7. Activate virtualenv
pyenv activate cnn_env

# 8. Upgrade pip inside virtualenv
pip install --upgrade pip

# 9. Install tensorflow 1.15 and other dependencies
pip install tensorflow==1.15

# 10. Optional: install dependencies from requirements.txt if file exists
#if [ -f "requirements.txt" ]; then
#    pip install -r requirements.txt
#fi

echo ""
echo "✅ Setup complete!"
echo "To activate later, run:"
echo "    pyenv activate cnn_env"
echo "Or restart your terminal to auto-load pyenv."

