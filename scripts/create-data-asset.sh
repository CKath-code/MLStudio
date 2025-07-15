#!/bin/bash

# Script to create a data asset with a unique timestamp-based name
# This prevents archived container errors

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
UNIQUE_NAME="used-cars-data-${TIMESTAMP}"

echo "Creating data asset with unique name: ${UNIQUE_NAME}"

# Create data asset with unique name
az ml data create \
  --file mlops/azureml/train/data.yml \
  --name "${UNIQUE_NAME}" \
  --resource-group defalt_resource_group \
  --workspace-name FitwellWorkspace

if [ $? -eq 0 ]; then
    echo "✅ Data asset created successfully with name: ${UNIQUE_NAME}"
    echo "📝 Remember to update your pipeline files to reference: azureml:${UNIQUE_NAME}@latest"
else
    echo "❌ Failed to create data asset"
    exit 1
fi
