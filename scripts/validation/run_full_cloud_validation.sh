#!/bin/bash
# TorchBridge v0.4.34 Full Cloud Validation
#
# This script orchestrates validation across all cloud providers and backends.
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - GCP CLI configured (gcloud auth login)
#   - Intel DevCloud SSH key configured
#
# Usage:
#   ./run_full_cloud_validation.sh [--free-only] [--spot-only] [--full]
#
# Cost Estimates:
#   --free-only: $0 (Intel DevCloud, Colab notebooks)
#   --spot-only: ~$50 (spot/preemptible instances)
#   --full: ~$150 (includes A100/H100 premium instances)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/reports/cloud_validation"
RESULTS_FILE="$REPORTS_DIR/validation_results.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
MODE="free-only"
while [[ $# -gt 0 ]]; do
    case $1 in
        --free-only) MODE="free-only"; shift ;;
        --spot-only) MODE="spot-only"; shift ;;
        --full) MODE="full"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}TorchBridge v0.4.34 Full Cloud Validation${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "Mode: ${YELLOW}$MODE${NC}"
echo -e "Reports: $REPORTS_DIR"
echo ""

# Initialize results
mkdir -p "$REPORTS_DIR"
echo '{"validation_start": "'$(date -Iseconds)'", "mode": "'$MODE'", "results": {}}' > "$RESULTS_FILE"

# Function to update results
update_result() {
    local provider=$1
    local instance=$2
    local status=$3
    local details=$4

    # Use Python to update JSON (portable)
    python3 << EOF
import json
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)
data['results']['${provider}_${instance}'] = {
    'status': '$status',
    'details': '$details',
    'timestamp': '$(date -Iseconds)'
}
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
EOF
}

# ============================================================
# TIER 1: FREE RESOURCES (Always run)
# ============================================================

echo -e "\n${GREEN}=== TIER 1: Free Resources ===${NC}\n"

# Intel DevCloud (Free)
echo -e "${BLUE}[1/4] Intel DevCloud (Arc A770)${NC}"
if command -v ssh &> /dev/null; then
    echo "  Checking Intel DevCloud connectivity..."
    if ssh -o BatchMode=yes -o ConnectTimeout=5 devcloud echo "connected" 2>/dev/null; then
        echo -e "  ${GREEN}✓ Connected to Intel DevCloud${NC}"
        echo "  Submitting validation job..."

        # Copy validation script
        scp "$REPORTS_DIR/validation_arc-a770.sh" devcloud:~/torchbridge_validation.sh 2>/dev/null || true

        # Submit job
        JOB_ID=$(ssh devcloud "qsub -l nodes=1:gpu:ppn=2 -d . ~/torchbridge_validation.sh" 2>/dev/null) || JOB_ID="failed"

        if [[ "$JOB_ID" != "failed" ]]; then
            echo -e "  ${GREEN}✓ Job submitted: $JOB_ID${NC}"
            update_result "intel" "arc-a770" "submitted" "Job ID: $JOB_ID"
        else
            echo -e "  ${YELLOW}⚠ Job submission failed (may need interactive session)${NC}"
            update_result "intel" "arc-a770" "manual_required" "Use interactive qsub"
        fi
    else
        echo -e "  ${YELLOW}⚠ Intel DevCloud not configured${NC}"
        echo "  To configure: https://devcloud.intel.com/oneapi/get_started/"
        update_result "intel" "arc-a770" "skipped" "DevCloud not configured"
    fi
else
    echo -e "  ${YELLOW}⚠ SSH not available${NC}"
    update_result "intel" "arc-a770" "skipped" "SSH not available"
fi

# Google Colab (Free - Manual)
echo -e "\n${BLUE}[2/4] Google Colab (T4 GPU)${NC}"
echo "  Notebook generated: $REPORTS_DIR/validation_google_colab_free_t4.ipynb"
echo "  Manual steps required:"
echo "    1. Open https://colab.research.google.com"
echo "    2. Upload the notebook"
echo "    3. Runtime > Change runtime type > GPU"
echo "    4. Run all cells"
update_result "colab" "t4" "manual_required" "See notebook"

# Kaggle (Free - Manual)
echo -e "\n${BLUE}[3/4] Kaggle Notebooks (P100/T4)${NC}"
echo "  Notebook generated: $REPORTS_DIR/validation_kaggle_notebooks_p100_t4.ipynb"
echo "  Manual steps required:"
echo "    1. Open https://www.kaggle.com/code"
echo "    2. New Notebook > File > Import Notebook"
echo "    3. Settings > Accelerator > GPU"
echo "    4. Run all cells"
update_result "kaggle" "p100" "manual_required" "See notebook"

# Lightning.ai (Free - Manual)
echo -e "\n${BLUE}[4/4] Lightning.ai (T4 GPU)${NC}"
echo "  Notebook generated: $REPORTS_DIR/validation_lightning.ai_t4.ipynb"
echo "  Manual steps required:"
echo "    1. Open https://lightning.ai"
echo "    2. Create new Studio with GPU"
echo "    3. Upload and run notebook"
update_result "lightning" "t4" "manual_required" "See notebook"

if [[ "$MODE" == "free-only" ]]; then
    echo -e "\n${GREEN}=== Free-tier validation complete ===${NC}"
    echo "Results saved to: $RESULTS_FILE"
    exit 0
fi

# ============================================================
# TIER 2: SPOT/PREEMPTIBLE INSTANCES (~$50)
# ============================================================

echo -e "\n${GREEN}=== TIER 2: Spot/Preemptible Instances ===${NC}\n"

# AWS g5.xlarge Spot (A10G - ~$0.30/hr)
echo -e "${BLUE}[1/3] AWS g5.xlarge Spot (A10G)${NC}"
if command -v aws &> /dev/null && aws sts get-caller-identity &> /dev/null; then
    echo "  AWS CLI configured, launching spot instance..."

    # Launch spot instance
    SPOT_REQUEST=$(aws ec2 request-spot-instances \
        --spot-price "0.50" \
        --instance-count 1 \
        --type "one-time" \
        --launch-specification '{
            "ImageId": "ami-0c55b159cbfafe1f0",
            "InstanceType": "g5.xlarge",
            "KeyName": "torchbridge-validation"
        }' \
        --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
        --output text 2>/dev/null) || SPOT_REQUEST="failed"

    if [[ "$SPOT_REQUEST" != "failed" && "$SPOT_REQUEST" != "" ]]; then
        echo -e "  ${GREEN}✓ Spot request: $SPOT_REQUEST${NC}"
        update_result "aws" "g5.xlarge" "spot_requested" "Request: $SPOT_REQUEST"
    else
        echo -e "  ${YELLOW}⚠ Spot request failed (check AWS quotas)${NC}"
        update_result "aws" "g5.xlarge" "failed" "Spot request failed"
    fi
else
    echo -e "  ${YELLOW}⚠ AWS CLI not configured${NC}"
    echo "  Run: aws configure"
    update_result "aws" "g5.xlarge" "skipped" "AWS not configured"
fi

# GCP g2-standard-4 Preemptible (L4 - ~$0.25/hr)
echo -e "\n${BLUE}[2/3] GCP g2-standard-4 Preemptible (L4)${NC}"
if command -v gcloud &> /dev/null && gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "  GCP CLI configured, launching preemptible instance..."

    INSTANCE_NAME="torchbridge-val-$(date +%s)"
    gcloud compute instances create "$INSTANCE_NAME" \
        --zone=us-central1-a \
        --machine-type=g2-standard-4 \
        --accelerator=type=nvidia-l4,count=1 \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --preemptible \
        --metadata-from-file=startup-script="$REPORTS_DIR/validation_g2-standard-4.sh" \
        2>/dev/null && {
            echo -e "  ${GREEN}✓ Instance created: $INSTANCE_NAME${NC}"
            update_result "gcp" "g2-standard-4" "running" "Instance: $INSTANCE_NAME"
        } || {
            echo -e "  ${YELLOW}⚠ Instance creation failed (check quotas)${NC}"
            update_result "gcp" "g2-standard-4" "failed" "Creation failed"
        }
else
    echo -e "  ${YELLOW}⚠ GCP CLI not configured${NC}"
    echo "  Run: gcloud auth login"
    update_result "gcp" "g2-standard-4" "skipped" "GCP not configured"
fi

# Lambda Labs (on-demand ~$0.60/hr)
echo -e "\n${BLUE}[3/3] Lambda Labs (A10)${NC}"
echo "  Lambda Labs requires manual API setup"
echo "  Visit: https://cloud.lambdalabs.com"
update_result "lambda" "a10" "manual_required" "API setup needed"

if [[ "$MODE" == "spot-only" ]]; then
    echo -e "\n${GREEN}=== Spot-tier validation complete ===${NC}"
    echo "Results saved to: $RESULTS_FILE"
    exit 0
fi

# ============================================================
# TIER 3: PREMIUM INSTANCES (~$100 additional)
# ============================================================

echo -e "\n${GREEN}=== TIER 3: Premium Instances ===${NC}\n"

# AWS p4d.24xlarge Spot (8x A100)
echo -e "${BLUE}[1/3] AWS p4d.24xlarge Spot (8x A100)${NC}"
if command -v aws &> /dev/null && aws sts get-caller-identity &> /dev/null; then
    echo "  Requesting 8x A100 spot instance (~\$9.80/hr spot)..."
    # Note: This requires significant AWS quotas
    echo -e "  ${YELLOW}⚠ Requires p4d quota increase${NC}"
    echo "  Script ready: $REPORTS_DIR/validation_p4d.24xlarge.sh"
    update_result "aws" "p4d.24xlarge" "manual_required" "Quota increase needed"
else
    update_result "aws" "p4d.24xlarge" "skipped" "AWS not configured"
fi

# GCP a2-highgpu-1g (A100)
echo -e "\n${BLUE}[2/3] GCP a2-highgpu-1g (A100)${NC}"
if command -v gcloud &> /dev/null; then
    echo "  Script ready: $REPORTS_DIR/validation_a2-highgpu-1g.sh"
    echo -e "  ${YELLOW}⚠ Requires A100 quota${NC}"
    update_result "gcp" "a2-highgpu-1g" "manual_required" "Quota increase needed"
else
    update_result "gcp" "a2-highgpu-1g" "skipped" "GCP not configured"
fi

# GCP TPU v5e-8
echo -e "\n${BLUE}[3/3] GCP TPU v5e-8${NC}"
if command -v gcloud &> /dev/null; then
    echo "  Script ready: $REPORTS_DIR/validation_v5e-8.sh"
    echo -e "  ${YELLOW}⚠ Requires TPU quota${NC}"
    update_result "gcp" "v5e-8" "manual_required" "TPU quota needed"
else
    update_result "gcp" "v5e-8" "skipped" "GCP not configured"
fi

# ============================================================
# SUMMARY
# ============================================================

echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}VALIDATION SUMMARY${NC}"
echo -e "${BLUE}============================================================${NC}"

# Count results
python3 << EOF
import json
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

results = data.get('results', {})
submitted = sum(1 for r in results.values() if r.get('status') in ['submitted', 'running', 'spot_requested'])
manual = sum(1 for r in results.values() if r.get('status') == 'manual_required')
skipped = sum(1 for r in results.values() if r.get('status') == 'skipped')
failed = sum(1 for r in results.values() if r.get('status') == 'failed')

print(f"  Submitted/Running: {submitted}")
print(f"  Manual Required:   {manual}")
print(f"  Skipped:          {skipped}")
print(f"  Failed:           {failed}")
print(f"  Total:            {len(results)}")

# Finalize
data['validation_end'] = '$(date -Iseconds)'
data['summary'] = {
    'submitted': submitted,
    'manual_required': manual,
    'skipped': skipped,
    'failed': failed,
    'total': len(results)
}
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
EOF

echo -e "\nResults saved to: ${GREEN}$RESULTS_FILE${NC}"
echo -e "\n${YELLOW}Next Steps:${NC}"
echo "  1. Complete manual validations (Colab, Kaggle, Lightning.ai)"
echo "  2. Check spot instance status: aws ec2 describe-spot-instance-requests"
echo "  3. Check GCP instances: gcloud compute instances list"
echo "  4. Collect results when complete"
echo ""
echo -e "${GREEN}Validation orchestration complete!${NC}"
