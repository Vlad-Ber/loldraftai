#!/bin/bash

# Script to switch between development and production API key configurations
# Usage: ./scripts/switch-api-key.sh [development|production]

API_KEY_TYPE=${1:-development}

if [[ "$API_KEY_TYPE" != "development" && "$API_KEY_TYPE" != "production" ]]; then
    echo "Usage: $0 [development|production]"
    echo "  development: Use development API key limits (20 req/s, 100 req/2min)"
    echo "  production:  Use production API key limits (500 req/s, 30k req/10min)"
    exit 1
fi

echo "Switching to $API_KEY_TYPE API key configuration..."

# Set environment variable
export API_KEY_TYPE=$(echo $API_KEY_TYPE | tr '[:lower:]' '[:upper:]')

echo "API_KEY_TYPE set to: $API_KEY_TYPE"

# Show current configuration
echo ""
echo "Current rate limits:"
if [[ "$API_KEY_TYPE" == "DEVELOPMENT" ]]; then
    echo "- 20 requests per second"
    echo "- 100 requests per 2 minutes"
    echo "- 7 total services running"
    echo "- Per-service: ~2.3 requests/second (with 80% safety margin)"
elif [[ "$API_KEY_TYPE" == "PRODUCTION" ]]; then
    echo "- 500 requests per second"
    echo "- 30,000 requests per 10 minutes"
    echo "- 7 total services running"
    echo "- Per-service: ~57 requests/second (with 80% safety margin)"
fi

echo ""
echo "To apply this configuration, restart your PM2 services:"
echo "pm2 restart all" 