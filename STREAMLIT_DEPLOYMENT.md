# AdaptNav Streamlit Deployment Guide

This guide will help you deploy the AdaptNav demo as a web application using Streamlit.

## 🚀 Quick Deploy to Streamlit Cloud (Recommended)

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

### Steps

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Add Streamlit web demo"
   git push origin main
   ```

2. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

3. **Deploy your app**:
   - Click "New app"
   - Select your repository: `SoriaAbd/Hack`
   - Set the main file path: `streamlit_app.py`
   - Set the Python version: `3.9` or higher
   - Click "Deploy"

4. **Wait for deployment** (usually 2-5 minutes)

5. **Get your URL**:
   - Your app will be available at: `https://[your-app-name].streamlit.app`
   - Share this URL for your hackathon submission!

## 🏃 Run Locally

### Option 1: Using the existing environment

```bash
# Install Streamlit
pip install streamlit

# Run the app
streamlit run streamlit_app.py
```

### Option 2: Using a fresh environment

```bash
# Create virtual environment
python -m venv streamlit_env

# Activate it
# On Windows:
streamlit_env\Scripts\activate
# On Mac/Linux:
source streamlit_env/bin/activate

# Install dependencies
pip install -r streamlit_requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Alternative Deployment Options

### Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose "Streamlit" as the SDK
3. Upload your files:
   - `streamlit_app.py`
   - `streamlit_requirements.txt`
   - `adaptnav/` folder (entire package)
4. Your app will be live at: `https://huggingface.co/spaces/[username]/[space-name]`

### Heroku

```bash
# Install Heroku CLI
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=$PORT" > Procfile

# Create setup.sh
cat > setup.sh << 'EOF'
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
EOF

# Deploy
heroku create your-app-name
git push heroku main
```

### Google Cloud Run

```bash
# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r streamlit_requirements.txt
EXPOSE 8080
CMD streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0
EOF

# Build and deploy
gcloud builds submit --tag gcr.io/[PROJECT-ID]/adaptnav
gcloud run deploy --image gcr.io/[PROJECT-ID]/adaptnav --platform managed
```

## 🎮 Using the Web Demo

### Controls

1. **Initialize Simulation**: Click to create a new simulation instance
2. **Step**: Execute one simulation step (0.1 seconds)
3. **Run 10**: Execute 10 steps quickly
4. **Auto-run**: Toggle continuous simulation

### Features

- **Live Visualization**: See the robot navigate in real-time
- **Metrics Dashboard**: Track position, velocity, and progress
- **Interactive Controls**: Step through or auto-run the simulation
- **Responsive Design**: Works on desktop and mobile

### What You'll See

- 🔵 Blue circle: Robot with orientation arrow
- 🟢 Green circle: Goal position
- 🟠 Orange circles: Workers (dynamic obstacles)
- 🔴 Red circle: Forklift (larger obstacle)
- ⬜ Gray rectangles: Warehouse shelves (static obstacles)
- 🟢 Green dashed line: Planned path
- 🔴 Red dashed circle: Safety zone

## 🐛 Troubleshooting

### "Module not found" errors

Make sure all AdaptNav components are in the `adaptnav/` folder and properly structured.

### Slow performance

The web version updates on each interaction. Use "Run 10" for faster progress or enable "Auto-run" for continuous simulation.

### Deployment fails

Check that:
- `streamlit_requirements.txt` is in the root directory
- `streamlit_app.py` is in the root directory
- All imports are available
- Python version is 3.8 or higher

## 📝 Configuration

### Streamlit Cloud Settings

In your Streamlit Cloud dashboard, you can configure:
- **Secrets**: Add API keys or configuration
- **Resources**: Adjust memory/CPU allocation
- **Python version**: Set to 3.9 or higher
- **Requirements**: Automatically detected from `streamlit_requirements.txt`

### Custom Domain

After deployment, you can:
1. Go to your app settings
2. Add a custom domain
3. Update DNS records as instructed

## 🎯 For Hackathon Submission

Your demo URL will be:
```
https://[your-app-name].streamlit.app
```

Or if using Hugging Face:
```
https://huggingface.co/spaces/[username]/adaptnav-demo
```

Make sure to:
- ✅ Test the URL before submitting
- ✅ Ensure the app loads and runs properly
- ✅ Add a description in your Streamlit app settings
- ✅ Include screenshots in your README

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Deployment Tutorials](https://docs.streamlit.io/streamlit-community-cloud/get-started)

## 🆘 Need Help?

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Test locally first with `streamlit run streamlit_app.py`
3. Verify all dependencies are in `streamlit_requirements.txt`
4. Check that the `adaptnav` package is properly structured

---

**Ready to deploy?** Follow the Quick Deploy steps above and you'll have a live demo in minutes! 🚀
