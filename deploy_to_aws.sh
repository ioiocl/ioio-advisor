#!/bin/bash
set -e

# ====== USER VARIABLES ======
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="<your-aws-account-id>"  # <-- CHANGE THIS
EKS_CLUSTER_NAME="my-finance-cluster"   # <-- CHANGE THIS

# ====== AGENT NAMES ======
AGENTS=(intention retriever reasoner writer designer)

# ====== 1. Authenticate AWS CLI (assumes aws configure already run) ======
echo "\n[1/8] Checking AWS CLI authentication..."
aws sts get-caller-identity

# ====== 2. Create ECR repositories (idempotent) ======
echo "\n[2/8] Creating ECR repositories..."
for AGENT in "${AGENTS[@]}"; do
  aws ecr describe-repositories --repository-names ${AGENT}-agent --region $AWS_REGION || \
  aws ecr create-repository --repository-name ${AGENT}-agent --region $AWS_REGION
  echo "ECR repo ensured: ${AGENT}-agent"
done

# ====== 3. Authenticate Docker to ECR ======
echo "\n[3/8] Authenticating Docker to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# ====== 4. Build, tag, and push Docker images ======
echo "\n[4/8] Building, tagging, and pushing Docker images..."
for AGENT in "${AGENTS[@]}"; do
  IMAGE_NAME=${AGENT}-agent:latest
  ECR_IMAGE=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${AGENT}-agent:latest
  DOCKERFILE=Dockerfile.${AGENT}
  echo "Building $IMAGE_NAME with $DOCKERFILE..."
  docker build -f $DOCKERFILE -t $IMAGE_NAME .
  docker tag $IMAGE_NAME $ECR_IMAGE
  docker push $ECR_IMAGE
done

# ====== 5. Update kubeconfig for EKS ======
echo "\n[5/8] Updating kubeconfig for EKS cluster..."
aws eks update-kubeconfig --region $AWS_REGION --name $EKS_CLUSTER_NAME

# ====== 6. Create Kubernetes secrets (idempotent) ======
echo "\n[6/8] Creating Kubernetes secrets..."
kubectl delete secret postgres-secret --ignore-not-found
kubectl create secret generic postgres-secret \
  --from-literal=POSTGRES_PASSWORD=YourStrongPassword  # <-- CHANGE THIS

kubectl delete secret api-keys --ignore-not-found
kubectl create secret generic api-keys \
  --from-literal=OPENAI_API_KEY=your-openai-key \
  --from-literal=STABILITY_API_KEY=your-stability-key
# Add more --from-literal as needed

echo "Secrets created."

# ====== 7. Deploy Kubernetes manifests ======
echo "\n[7/8] Deploying agent microservices to EKS..."
K8S_DIR="src/infrastructure/k8s"
for AGENT in "${AGENTS[@]}"; do
  # Patch image in deployment YAML (in-memory, not in file)
  DEPLOY_FILE="$K8S_DIR/${AGENT}-agent-deployment.yaml"
  kubectl apply -f <(\
    sed "s|image:.*|image: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${AGENT}-agent:latest|" $DEPLOY_FILE)
done

echo "\n[8/8] Done! Microservices deployed."
echo "Check with: kubectl get pods && kubectl get svc"
