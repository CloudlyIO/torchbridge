#!/usr/bin/env bash
# cloud_check.sh — Full sweep of all cloud resources that could be billing.
# Run BEFORE and AFTER any cloud validation session.
# Usage: bash scripts/cloud/cloud_check.sh
# Exit code: 0 = all clear, 1 = resources found (take action before proceeding)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FOUND=0

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  TorchBridge Cloud Resource Check — $(date '+%Y-%m-%d %H:%M')"
echo "════════════════════════════════════════════════════════════"

# ── AWS ──────────────────────────────────────────────────────────
echo ""
echo "[ AWS ] Scanning 15 regions for running/stopped instances..."

AWS_REGIONS="us-east-1 us-east-2 us-west-1 us-west-2 eu-west-1 eu-west-2 eu-west-3 eu-central-1 ap-southeast-1 ap-southeast-2 ap-northeast-1 ap-northeast-2 ap-south-1 sa-east-1 ca-central-1"

for region in $AWS_REGIONS; do
    result=$(aws ec2 describe-instances \
        --region "$region" \
        --filters "Name=instance-state-name,Values=running,pending,stopping,stopped" \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' \
        --output text 2>/dev/null || true)
    if [ -n "$result" ]; then
        echo -e "${RED}  !!! INSTANCES FOUND in $region !!!${NC}"
        echo "$result" | while read -r line; do echo "      $line"; done
        FOUND=1
    fi
done

# AWS orphaned EBS volumes
for region in us-east-1 us-west-2; do
    vols=$(aws ec2 describe-volumes \
        --region "$region" \
        --filters "Name=status,Values=available" \
        --query 'Volumes[].[VolumeId,Size]' \
        --output text 2>/dev/null || true)
    if [ -n "$vols" ]; then
        echo -e "${YELLOW}  Orphaned EBS volumes in $region:${NC}"
        echo "$vols" | while read -r line; do echo "    $line"; done
        FOUND=1
    fi
done

# AWS Elastic IPs (billed when unattached)
for region in us-east-1 us-west-2; do
    eips=$(aws ec2 describe-addresses \
        --region "$region" \
        --query 'Addresses[?!AssociationId].[AllocationId,PublicIp]' \
        --output text 2>/dev/null || true)
    if [ -n "$eips" ]; then
        echo -e "${YELLOW}  Unattached Elastic IPs in $region:${NC}"
        echo "$eips" | while read -r line; do echo "    $line"; done
        FOUND=1
    fi
done

# ── GCP ──────────────────────────────────────────────────────────
echo ""
echo "[ GCP ] Scanning project shahmod-kernel-pytorch..."

# Compute instances
gcp_vms=$(gcloud compute instances list \
    --project=shahmod-kernel-pytorch \
    --format="value(name,zone,status,machineType)" 2>/dev/null || true)
if [ -n "$gcp_vms" ]; then
    echo -e "${RED}  !!! GCP VMs RUNNING !!!${NC}"
    echo "$gcp_vms" | while read -r line; do echo "    $line"; done
    FOUND=1
fi

# TPU VMs — check all zones that have ever been used
TPU_ZONES="us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d us-east4-a us-east4-b us-east4-c us-west1-a us-west4-a us-west4-b us-west4-c europe-west4-a europe-west4-b europe-west4-c"
for zone in $TPU_ZONES; do
    tpu=$(gcloud compute tpus tpu-vm list \
        --project=shahmod-kernel-pytorch \
        --zone="$zone" \
        --format="value(name,state)" 2>/dev/null || true)
    if [ -n "$tpu" ]; then
        echo -e "${RED}  !!! TPU FOUND in $zone: $tpu !!!${NC}"
        FOUND=1
    fi
done

# GCP static addresses (IN_USE means a TPU/VM is attached)
addrs=$(gcloud compute addresses list \
    --project=shahmod-kernel-pytorch \
    --format="value(name,region,status)" 2>/dev/null || true)
if [ -n "$addrs" ]; then
    echo -e "${RED}  !!! GCP ADDRESSES (TPU/VM still attached) !!!${NC}"
    echo "$addrs" | while read -r line; do echo "    $line"; done
    FOUND=1
fi

# GCP persistent disks
disks=$(gcloud compute disks list \
    --project=shahmod-kernel-pytorch \
    --format="value(name,zone,sizeGb,status)" 2>/dev/null || true)
if [ -n "$disks" ]; then
    echo -e "${YELLOW}  GCP disks (verify these are intentional):${NC}"
    echo "$disks" | while read -r line; do echo "    $line"; done
    # Disks alone are low cost but flag them
fi

# ── RunPod ───────────────────────────────────────────────────────
echo ""
echo "[ RunPod ] Checking pods..."

runpod_pods=$(runpodctl get pod 2>/dev/null | tail -n +2 || true)
if [ -n "$runpod_pods" ]; then
    echo -e "${RED}  !!! RunPod PODS EXIST (EXITED pods still bill for storage) !!!${NC}"
    echo "$runpod_pods" | while read -r line; do echo "    $line"; done
    echo -e "${YELLOW}  Use: runpodctl remove pod POD_ID${NC}"
    FOUND=1
fi

# ── AMD ──────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[ AMD ] CANNOT CHECK AUTOMATICALLY — MANUAL ACTION REQUIRED${NC}"
echo "  AMD Developer Cloud (~\$750/mo) has no CLI."
echo "  Verify manually: https://amd.com/developer/resources/ai-cloud.html"
echo "  poweroff via SSH does NOT stop billing. Only portal destruction does."

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
if [ "$FOUND" -eq 0 ]; then
    echo -e "  ${GREEN}✓ ALL CLEAR — No billable cloud resources found${NC}"
    echo ""
    exit 0
else
    echo -e "  ${RED}✗ RESOURCES FOUND — TERMINATE BEFORE PROCEEDING${NC}"
    echo ""
    echo "  Quick terminate commands:"
    echo "    AWS:    aws ec2 terminate-instances --region REGION --instance-ids ID"
    echo "    GCP VM: gcloud compute instances delete NAME --zone=ZONE --project=shahmod-kernel-pytorch --quiet"
    echo "    GCP TPU: gcloud compute tpus tpu-vm delete NAME --zone=ZONE --project=shahmod-kernel-pytorch --quiet"
    echo "    RunPod: runpodctl remove pod POD_ID"
    echo ""
    exit 1
fi
