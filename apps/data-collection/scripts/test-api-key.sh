#!/bin/bash

# Script to test API key and determine if it's production or development
# Usage: ./scripts/test-api-key.sh

echo "Testing Riot Games API key..."

# Load environment variables
source apps/data-collection/.env

# Check if API key is set
if [ -z "$X_RIOT_API_KEY" ]; then
    echo "❌ Error: X_RIOT_API_KEY is not set"
    exit 1
fi

echo "✅ API Key found: ${X_RIOT_API_KEY:0:20}..."

# Test the API key by making a request
echo "Testing API key with a simple request..."

# Make a test request to the Riot API
response=$(curl -s -w "%{http_code}" \
  -H "X-Riot-Token: $X_RIOT_API_KEY" \
  "https://euw1.api.riotgames.com/lol/summoner/v4/summoners/by-name/test" \
  -o /dev/null)

echo "Response code: $response"

if [ "$response" = "200" ]; then
    echo "✅ API key is valid!"
    
    # Check if it's a production key by looking at the format
    # Production keys typically start with RGAPI- and are longer
    if [[ "$X_RIOT_API_KEY" == RGAPI-* ]]; then
        echo "🔍 API Key Analysis:"
        echo "   Format: RGAPI- (Development/Personal API Key)"
        echo "   Type: Development API Key"
        echo "   Rate Limits: 20 requests/second, 100 requests/2 minutes"
        echo ""
        echo "💡 To use production limits, you would need a different API key format"
        echo "   Production keys typically have different prefixes and higher limits"
    else
        echo "🔍 API Key Analysis:"
        echo "   Format: Unknown (might be production)"
        echo "   Type: Unknown - test with production limits"
        echo "   Rate Limits: 500 requests/second, 30,000 requests/10 minutes"
    fi
else
    echo "❌ API key test failed with response code: $response"
    echo "   This could mean:"
    echo "   - Invalid API key"
    echo "   - Rate limit exceeded"
    echo "   - Network issues"
fi

echo ""
echo "📋 Current Configuration:"
echo "   API_KEY_TYPE: ${API_KEY_TYPE:-DEVELOPMENT}"
echo "   Total Services: 7"
echo "   Current Rate Limits: Conservative (safe for development key)"

echo ""
echo "🚀 To start services with current configuration:"
echo "   cd apps/data-collection"
echo "   pm2 start ecosystem.config.cjs"
echo ""
echo "🔄 To switch to production limits (if you have a production key):"
echo "   ./scripts/switch-api-key.sh production"
echo "   pm2 restart all" 