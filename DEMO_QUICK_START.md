# AdaptNav Web Demo - Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Install Streamlit
```bash
pip install streamlit
```

### Step 2: Run the Demo
```bash
streamlit run streamlit_app.py
```

### Step 3: Use the Demo
1. Click "🔄 Initialize Simulation"
2. Click "▶️ Step" to move the robot
3. Or enable "🔁 Auto-run" for continuous simulation

That's it! 🎉

## 🌐 Deploy to Web (10 Minutes)

### Quick Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit demo"
   git push
   ```

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repo and `streamlit_app.py`
   - Click "Deploy"

3. **Get Your URL**
   - Wait 2-5 minutes
   - Your demo will be live at: `https://[app-name].streamlit.app`
   - Share this URL! 🎯

## 🎮 Using the Demo

### Controls
- **Initialize Simulation**: Start a new simulation
- **Step**: Execute one time step (0.1s)
- **Run 10**: Execute 10 steps quickly
- **Auto-run**: Continuous simulation

### What You See
- 🔵 **Blue circle**: Robot (with direction arrow)
- 🟢 **Green circle**: Goal position
- 🟠 **Orange circles**: Workers (moving obstacles)
- 🔴 **Red circle**: Forklift (larger obstacle)
- ⬜ **Gray boxes**: Warehouse shelves
- 🟢 **Green dashed line**: Planned path
- 🔴 **Red dashed circle**: Safety zone

### Metrics
- **Position**: Current robot location (x, y)
- **Goal Distance**: How far from the goal
- **Velocity**: Current speed (m/s)
- **Simulation Time**: Elapsed time

## 🎯 For Hackathon Judges

### Key Features to Highlight

1. **Hybrid Navigation**
   - A* algorithm for global path planning
   - Real-time obstacle avoidance
   - Safety control system

2. **Interactive Visualization**
   - Live robot movement
   - Dynamic obstacles
   - Real-time metrics

3. **Practical Application**
   - Warehouse automation
   - Safe human-robot collaboration
   - Scalable architecture

### Demo Flow (2 minutes)

1. **Initialize** (10 seconds)
   - Click "Initialize Simulation"
   - Show the warehouse layout

2. **Explain** (30 seconds)
   - Point out robot, goal, obstacles
   - Mention A* path planning
   - Highlight safety zones

3. **Run** (60 seconds)
   - Enable "Auto-run"
   - Watch robot navigate
   - Point out obstacle avoidance
   - Show metrics updating

4. **Conclude** (20 seconds)
   - Robot reaches goal
   - Mention real-world applications
   - Invite questions

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install -r streamlit_requirements.txt
```

### App won't start
```bash
# Check if you're in the right directory
ls streamlit_app.py

# Try running with full path
python -m streamlit run streamlit_app.py
```

### Slow performance
- This is normal for the simulation
- Use "Run 10" for faster progress
- Or enable "Auto-run" and wait

### Deployment fails
- Verify `streamlit_requirements.txt` exists
- Check all files are committed to Git
- Ensure Python 3.8+ is specified

## 📚 More Information

- **Full Deployment Guide**: See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md)
- **Deployment Checklist**: See [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)
- **Project README**: See [README.md](README.md)
- **Test the App**: Run `python test_streamlit_app.py`

## 🆘 Need Help?

1. **Test locally first**: `streamlit run streamlit_app.py`
2. **Check the logs**: Look for error messages
3. **Verify dependencies**: `pip list | grep streamlit`
4. **Try a fresh install**: Create a new virtual environment

## ✅ Success Checklist

- [ ] Streamlit installed
- [ ] App runs locally
- [ ] Simulation initializes
- [ ] Robot moves and reaches goal
- [ ] Deployed to Streamlit Cloud (optional)
- [ ] Demo URL works and is shareable

---

**Ready to impress? Run the demo and show off AdaptNav! 🤖✨**
