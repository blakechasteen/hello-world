# AWS Quick Start - No Local GPU

## For Users Without NVIDIA GPU

**You have integrated graphics (11GB shared RAM)** → Use AWS for Week 2-4 testing.

**Cost**: ~$170 for 3 weeks (8 hours/day)
**Time**: 10 minutes setup

---

## Step 1: Create AWS Account (5 min)

1. Go to https://aws.amazon.com/
2. Click "Create an AWS Account"
3. Enter email, password, account name
4. Enter payment info (won't charge until you use resources)
5. Verify phone number
6. Choose "Basic Support" (free)

**Done!** You now have an AWS account.

---

## Step 2: Launch GPU Instance (3 min)

### 2.1: Go to EC2

1. Log into AWS Console: https://console.aws.amazon.com/
2. Search for "EC2" in top search bar
3. Click "EC2" (Virtual Servers in the Cloud)

### 2.2: Launch Instance

1. Click orange "Launch Instance" button
2. **Name**: `ouroboros-gpu`
3. **Application and OS Images**:
   - Click "Quick Start"
   - Select "Ubuntu"
   - Choose "Ubuntu Server 22.04 LTS"
4. **Instance type**:
   - Search for: `g5.xlarge`
   - Select it (A10G GPU, 24GB VRAM, $1.01/hr)
5. **Key pair**:
   - Click "Create new key pair"
   - Name: `ouroboros-key`
   - Type: RSA
   - Format: .pem
   - Click "Create key pair"
   - **Save the .pem file!** (You'll need it to SSH)
6. **Network settings**:
   - Click "Edit"
   - Check "Allow SSH from: My IP"
   - Check "Allow HTTP from: Anywhere"
   - Check "Allow HTTPS from: Anywhere"
7. **Configure storage**:
   - Change "8 GiB" to "100 GiB"
8. **Advanced details** → Skip
9. Click "Launch instance"

**Wait 2-3 minutes** for instance to start.

### 2.3: Get Instance IP

1. Click "Instances" in left sidebar
2. Find your instance "ouroboros-gpu"
3. Wait until "Instance state" = "Running"
4. Copy the "Public IPv4 address" (e.g., 54.123.45.67)

---

## Step 3: Connect via SSH (2 min)

### Windows (PowerShell)

```powershell
# Move key to safe location
mkdir C:\Users\YourName\.ssh
move Downloads\ouroboros-key.pem C:\Users\YourName\.ssh\

# Set permissions (important!)
icacls C:\Users\YourName\.ssh\ouroboros-key.pem /inheritance:r
icacls C:\Users\YourName\.ssh\ouroboros-key.pem /grant:r "%USERNAME%:R"

# Connect (replace with your IP)
ssh -i C:\Users\YourName\.ssh\ouroboros-key.pem ubuntu@54.123.45.67
```

**Type "yes"** when asked about fingerprint.

You're now connected to your AWS GPU instance! 🎉

---

## Step 4: Install Everything (15 min)

### 4.1: Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### 4.2: Install NVIDIA Drivers

```bash
# AWS g5 instances have drivers pre-installed
nvidia-smi

# Should show:
# GPU 0: NVIDIA A10G (24GB)
```

If `nvidia-smi` doesn't work:
```bash
sudo apt install -y nvidia-driver-525
sudo reboot
# Wait 2 min, then SSH back in
```

### 4.3: Install Python

```bash
sudo apt install -y python3.10 python3.10-venv python3-pip
```

### 4.4: Create Project

```bash
# Create directory
mkdir ~/ouroboros
cd ~/ouroboros

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should print: CUDA: True
```

### 4.5: Install vLLM

```bash
pip install vllm==0.2.7
pip install transformers==4.36.0
pip install huggingface-hub
pip install fastapi uvicorn
```

### 4.6: Download Llama-2-7b-hf

```bash
# Get Hugging Face token from: https://huggingface.co/settings/tokens
huggingface-cli login
# Paste your token

# Download model (13.5GB, takes ~5 min on AWS)
huggingface-cli download meta-llama/Llama-2-7b-hf \
  --local-dir ~/models/Llama-2-7b-hf \
  --local-dir-use-symlinks False
```

### 4.7: Copy Ouroboros Code

**From your Windows machine**:
```powershell
# In new PowerShell window (not SSH)
scp -i C:\Users\YourName\.ssh\ouroboros-key.pem -r C:\Users\blake\OneDrive\Documents\mythRL\ouroboros ubuntu@54.123.45.67:~/
```

**Or manually**: Copy files via SFTP (WinSCP, FileZilla)

---

## Step 5: Test vLLM (2 min)

```bash
cd ~/ouroboros
source venv/bin/activate

cat > test.py << 'EOF'
from vllm import LLM, SamplingParams
import time

llm = LLM(model="/home/ubuntu/models/Llama-2-7b-hf", dtype="float16")
params = SamplingParams(temperature=0.0, max_tokens=256)

print("Testing on AWS A10G...")
start = time.time()
outputs = llm.generate(["Check: warfarin + aspirin"], params)
latency = (time.time() - start) * 1000

print(f"\n{outputs[0].outputs[0].text}")
print(f"\nLatency: {latency:.1f} ms")
EOF

python test.py
```

**Expected output**:
```
Latency: 15-20 ms
Text: "This combination should be BLOCKED due to severe bleeding risk..."
```

---

## Step 6: Run Web UI (1 min)

```bash
# Start web UI
python web_ui.py
```

**Access from your Windows browser**:
```
http://54.123.45.67:8000
(Use your instance's IP)
```

If it doesn't load, add firewall rule:
1. AWS Console → EC2 → Instances
2. Click your instance
3. Click "Security" tab
4. Click security group link
5. "Edit inbound rules"
6. "Add rule": Custom TCP, Port 8000, Source: My IP
7. Save

---

## Managing Costs

### Turn Off When Not Using

**IMPORTANT**: EC2 charges by the hour, even when idle!

**Stop instance** (saves money):
```
AWS Console → EC2 → Instances
→ Select instance
→ Instance state → Stop
```

**Stopped instance cost**: $0.10/hour (storage only)
**Running instance cost**: $1.01/hour

**Start again**:
```
Instance state → Start
```

### Cost Calculator

| Usage | Hours/Day | Days | Cost |
|-------|-----------|------|------|
| **8 hours/day, 21 days** | 8 | 21 | **$170** |
| 4 hours/day, 21 days | 4 | 21 | $85 |
| 24/7, 30 days | 24 | 30 | $730 |

**Pro tip**: Stop instance every night, save 50%!

### Set Billing Alarm

1. AWS Console → CloudWatch → Alarms
2. Create alarm
3. Metric: Billing → Total Estimated Charge
4. Threshold: $200
5. Email: your@email.com
6. Create alarm

**You'll get email if cost exceeds $200** (safety net)

---

## Monitoring

### Check GPU Usage

```bash
watch -n 1 nvidia-smi

# Shows real-time:
# - GPU utilization (should be 80-100% during inference)
# - Memory usage (should be ~18GB / 24GB)
# - Temperature (should be 50-70°C)
```

### Check Costs

1. AWS Console → Billing Dashboard
2. Shows current month charges
3. Updates every few hours

---

## After Week 4

### Option 1: Buy RTX 3090 (Best Long-Term)

**If you love the project**:
- Buy used RTX 3090 ($600-750)
- Set up at home (same instructions as mining rig guide)
- Stop AWS instance
- **Saves $8,000/year**

### Option 2: Stay on AWS

**If you prefer cloud**:
- Keep using g5.xlarge
- Cost: $730/month (24/7) or less (stop when not using)
- No hardware maintenance
- Professional infrastructure

### Option 3: Stop Everything

**If it didn't work out**:
- Terminate instance (EC2 → Terminate)
- Total cost: ~$170
- No ongoing charges
- Learned a lot!

---

## Troubleshooting

### Can't SSH

```powershell
# Check key permissions
icacls C:\Users\YourName\.ssh\ouroboros-key.pem

# Should show ONLY your username
# If not, remove inheritance:
icacls C:\Users\YourName\.ssh\ouroboros-key.pem /inheritance:r
icacls C:\Users\YourName\.ssh\ouroboros-key.pem /grant:r "%USERNAME%:R"
```

### vLLM Out of Memory

```python
# Reduce batch size
llm = LLM(
    model="/home/ubuntu/models/Llama-2-7b-hf",
    max_model_len=1024,  # Reduce from 2048
    gpu_memory_utilization=0.85  # Reduce from 0.9
)
```

### Can't Access Web UI

```bash
# Check if running
sudo netstat -tlnp | grep 8000

# Add firewall rule (see Step 6)
```

---

## Summary

**AWS g5.xlarge for Week 2-4**:
- ✅ Setup: 10 minutes
- ✅ Cost: $170 (8 hours/day, 3 weeks)
- ✅ Performance: 18ms latency (same as RTX 3090)
- ✅ Professional GPU (A10G 24GB)
- ✅ No hardware purchase
- ✅ Test everything
- ✅ Decide later if you want to buy GPU

**After Week 4**: Buy RTX 3090 ($700) if you want to continue long-term (saves $8,000/year).

**Ready to start?** 🚀
