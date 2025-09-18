# Data Management Guide

## 🚫 **Files NOT Tracked by Git**

The following files and directories are excluded from git to keep the repository lightweight and secure:

### **Large Data Files**
- `data/` - Raw PDF documents and case files (70MB+)
- `data_processing/output/` - Processed parquet files (3.3GB+)
- `data_processing/save/` - Additional data storage
- `*.parquet` - All parquet data files
- `*.pdf` - All PDF documents
- `*.zip`, `*.tar.gz`, `*.rar`, `*.7z` - Archive files

### **Sensitive Configuration**
- `config/firebase-credentials/` - Firebase service account JSON
- `*.json` - Configuration files with secrets
- `.env` - Environment variables

### **Generated Files**
- `__pycache__/` - Python cache directories
- `.ipynb_checkpoints/` - Jupyter notebook checkpoints
- `*.log` - Log files
- `*.tmp`, `*.temp` - Temporary files

## ✅ **Files Tracked by Git**

### **Source Code**
- `backend/` - All Python source code
- `frontend/` - HTML, CSS, JavaScript files
- `scripts/` - Utility scripts
- `*.py` - Python files
- `*.html`, `*.css`, `*.js` - Frontend files

### **Configuration (Non-sensitive)**
- `config/settings.py` - Application settings
- `config/__init__.py` - Package initialization
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration

### **Documentation**
- `*.md` - Markdown documentation
- `README.md` - Project documentation
- `CLOUD_RUN_DEPLOYMENT.md` - Deployment guide
- `FIRESTORE_SETUP.md` - Firestore setup guide

## 🔧 **Data File Management**

### **For Development**
1. **Data files are local only** - They won't be pushed to git
2. **Regenerate as needed** - Use the chunking scripts to recreate parquet files
3. **Keep data in `data/`** - Store raw PDFs and documents here

### **For Production**
1. **Data files in container** - Include in Docker image if needed
2. **Cloud storage** - Consider using Google Cloud Storage for large datasets
3. **Environment variables** - Use secrets for sensitive configuration

### **For Team Collaboration**
1. **Share data separately** - Use cloud storage or file sharing
2. **Document data sources** - Keep track of where data comes from
3. **Version control** - Use data versioning tools if needed

## 📁 **Directory Structure**

```
icc_chatbot/
├── data/                          # ❌ NOT in git (70MB+)
│   └── AI IHL/                   # PDF documents
├── data_processing/
│   ├── output/                   # ❌ NOT in git (3.3GB+)
│   │   └── *.parquet            # Processed data
│   ├── chunking/                 # ✅ In git
│   │   └── *.py                 # Processing scripts
│   └── notebooks/                # ✅ In git
│       └── *.ipynb              # Analysis notebooks
├── config/
│   ├── settings.py               # ✅ In git
│   ├── __init__.py              # ✅ In git
│   └── firebase-credentials/    # ❌ NOT in git
│       └── *.json               # Service account
├── backend/                      # ✅ In git
├── frontend/                     # ✅ In git
└── scripts/                      # ✅ In git
```

## 🚀 **Deployment Considerations**

### **Docker Build**
- Data files are included in the container
- Service account JSON is included for Firestore access
- Large files increase build time and image size

### **Cloud Run**
- Container includes all necessary files
- No need to mount external volumes for data
- Consider using Cloud Storage for very large datasets

## 🔒 **Security Best Practices**

1. **Never commit secrets** - Use environment variables
2. **Use .gitignore** - Keep sensitive files out of git
3. **Regular audits** - Check what's being tracked
4. **Team guidelines** - Ensure everyone follows the same practices

## 📊 **File Size Limits**

- **GitHub**: 100MB per file, 1GB per repository
- **Git LFS**: For files over 100MB
- **Cloud Run**: 10GB container size limit
- **Best Practice**: Keep repository under 100MB

## 🛠️ **Commands for Data Management**

```bash
# Check what's being tracked
git ls-files | grep -E "\.(parquet|pdf)$"

# Check file sizes
du -sh data/ data_processing/output/

# Remove large files from tracking
git rm --cached large-file.parquet

# Check git status
git status

# See what's ignored
git status --ignored
```

## 📝 **Notes**

- Data files are essential for the application but too large for git
- The application can regenerate processed data from raw sources
- Configuration files are tracked but sensitive data is excluded
- This setup keeps the repository lightweight and secure
