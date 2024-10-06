#create virtual env
python3 -m venv .venv

#activate virtual env
source .venv/bin/activate

#install requirements
pip install -r requirements.txt

# adds the project root to the python path, in any python 3 installation
SITE_PACKAGES_DIR=$(find $VIRTUAL_ENV/lib -name site-packages)
PROJECT_PTH="$SITE_PACKAGES_DIR/project.pth"
mkdir -p "$(dirname "$PROJECT_PTH")"
echo $PWD > "$PROJECT_PTH"
